import { ModelConfig } from "../types";

export const getEnsembleSuggestion = async (_models: ModelConfig[], _currentAccuracy: number): Promise<string> => {
  return "AI suggestion unavailable.";
};
