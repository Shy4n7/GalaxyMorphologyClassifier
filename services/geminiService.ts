
import { GoogleGenAI, Type } from "@google/genai";
import { ModelConfig } from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });

export const getEnsembleSuggestion = async (models: ModelConfig[], currentAccuracy: number) => {
  try {
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: `I am running an ensemble of these models: ${models.map(m => m.name).join(', ')}. 
      Their current weights are ${models.map(m => m.weight).join(', ')}. 
      Total ensemble confidence is ${currentAccuracy}%. 
      Suggest optimized weights to increase overall accuracy. Return the answer in short, bulleted technical points.`,
      config: {
        temperature: 0.7,
        topP: 0.9,
      }
    });

    return response.text;
  } catch (error) {
    console.error("Gemini Error:", error);
    return "AI suggestion unavailable. Please check connectivity.";
  }
};
