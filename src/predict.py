"""
CLI Tool for Galaxy Classification
Usage: python predict.py <image_path>
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Galaxy class names
CLASS_NAMES = [
    "Disturbed Galaxies",
    "Merging Galaxies", 
    "Round Smooth Galaxies",
    "In-between Round Smooth Galaxies",
    "Cigar Shaped Smooth Galaxies",
    "Barred Spiral Galaxies",
    "Unbarred Tight Spiral Galaxies",
    "Unbarred Loose Spiral Galaxies",
    "Edge-on Galaxies without Bulge",
    "Edge-on Galaxies with Bulge"
]

# Model architectures (same as training)
class ResNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.base(x)

class DenseNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = self.base.classifier.in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.base(x)

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=10, version='b0'):
        super().__init__()
        if version == 'b0':
            self.base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = 1280
        else:  # b2
            self.base = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
            in_features = 1408
        
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.35 if version == 'b0' else 0.5),
            nn.Linear(in_features, 384 if version == 'b0' else 768),
            nn.ReLU(),
            nn.BatchNorm1d(384 if version == 'b0' else 768),
            nn.Dropout(0.3 if version == 'b0' else 0.45),
            nn.Linear(384 if version == 'b0' else 768, 192 if version == 'b0' else 384),
            nn.ReLU(),
            nn.BatchNorm1d(192 if version == 'b0' else 384),
            nn.Dropout(0.25 if version == 'b0' else 0.35),
            nn.Linear(192 if version == 'b0' else 384, num_classes)
        )
    
    def forward(self, x):
        return self.base(x)

def load_models(device):
    """Load all trained ensemble models"""
    models_dict = {}
    models_dir = 'models'
    
    model_configs = [
        ('ResNet-50', ResNetModel(), 'orchestrated_resnet50.pth'),
        ('DenseNet-121', DenseNetModel(), 'orchestrated_densenet121.pth'),
        ('EfficientNet-B0 V1', EfficientNetModel(version='b0'), 'orchestrated_efficientnet_b0_v1.pth'),
        ('EfficientNet-B0 V2', EfficientNetModel(version='b0'), 'orchestrated_efficientnet_b0_v2.pth'),
    ]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Loading models...", total=len(model_configs))
        
        for name, model, filename in model_configs:
            path = os.path.join(models_dir, filename)
            if os.path.exists(path):
                try:
                    state_dict = torch.load(path, map_location=device)
                    model.load_state_dict(state_dict)
                    model.to(device)
                    model.eval()
                    models_dict[name] = model
                    progress.update(task, advance=1, description=f"[green]✓ Loaded {name}")
                except Exception as e:
                    progress.update(task, advance=1, description=f"[red]✗ Failed {name}")
            else:
                progress.update(task, advance=1, description=f"[yellow]✗ Not found {name}")
    
    return models_dict

def preprocess_image(image_path, device):
    """Preprocess image for model inference"""
    img = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((69, 69)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor.to(device)

def predict(models_dict, image_tensor):
    """Run ensemble prediction"""
    predictions = []
    model_results = {}
    
    with torch.no_grad():
        for name, model in models_dict.items():
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            predictions.append(probs)
            
            pred_class = np.argmax(probs)
            model_results[name] = {
                'class': CLASS_NAMES[pred_class],
                'confidence': probs[pred_class] * 100
            }
    
    # Ensemble averaging
    ensemble_probs = np.mean(predictions, axis=0)
    pred_class = np.argmax(ensemble_probs)
    
    return {
        'prediction': CLASS_NAMES[pred_class],
        'confidence': ensemble_probs[pred_class] * 100,
        'ensemble_probs': ensemble_probs,
        'model_results': model_results
    }

def display_results(result, image_path):
    """Display results in a beautiful format"""
    # Main prediction panel
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]{result['prediction']}[/bold cyan]\n"
        f"[green]Confidence: {result['confidence']:.2f}%[/green]",
        title="[bold]🌌 Galaxy Classification Result[/bold]",
        border_style="cyan"
    ))
    
    # Top 3 predictions
    console.print("\n[bold]Top 3 Predictions:[/bold]")
    top3_indices = np.argsort(result['ensemble_probs'])[-3:][::-1]
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Galaxy Type", style="cyan")
    table.add_column("Confidence", justify="right", style="green")
    
    for i, idx in enumerate(top3_indices, 1):
        table.add_row(
            f"#{i}",
            CLASS_NAMES[idx],
            f"{result['ensemble_probs'][idx] * 100:.2f}%"
        )
    
    console.print(table)
    
    # Individual model predictions
    console.print("\n[bold]Individual Model Predictions:[/bold]")
    
    model_table = Table(show_header=True, header_style="bold yellow")
    model_table.add_column("Model", style="yellow")
    model_table.add_column("Prediction", style="cyan")
    model_table.add_column("Confidence", justify="right", style="green")
    
    for name, pred in result['model_results'].items():
        model_table.add_row(
            name,
            pred['class'],
            f"{pred['confidence']:.2f}%"
        )
    
    console.print(model_table)
    console.print()

def main():
    parser = argparse.ArgumentParser(
        description='Galaxy Morphology Classifier - CLI Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py galaxy.jpg
  python predict.py path/to/galaxy.png
  python predict.py galaxy.fits
        """
    )
    parser.add_argument('image', help='Path to galaxy image')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        console.print(f"[red]Error: Image file not found: {args.image}[/red]")
        sys.exit(1)
    
    # Setup device
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    
    console.print(Panel.fit(
        f"[bold]Galaxy Morphology Classifier[/bold]\n"
        f"Image: [cyan]{args.image}[/cyan]\n"
        f"Device: [green]{device}[/green]",
        border_style="blue"
    ))
    
    # Load models
    models_dict = load_models(device)
    
    if not models_dict:
        console.print("[red]Error: No models loaded. Please train models first.[/red]")
        sys.exit(1)
    
    # Preprocess image
    console.print("\n[cyan]Preprocessing image...[/cyan]")
    image_tensor = preprocess_image(args.image, device)
    
    # Predict
    console.print("[cyan]Running ensemble prediction...[/cyan]")
    result = predict(models_dict, image_tensor)
    
    # Display results
    display_results(result, args.image)

if __name__ == '__main__':
    main()
