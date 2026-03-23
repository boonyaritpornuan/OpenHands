import { useQuery } from "@tanstack/react-query";
import OptionService from "#/api/option-service/option-service.api";

const fetchAiConfigOptions = async () => {
  const [models, openRouterModels, agents, securityAnalyzers] =
    await Promise.all([
      OptionService.getModels(),
      OptionService.getOpenRouterModels(),
      OptionService.getAgents(),
      OptionService.getSecurityAnalyzers(),
    ]);

  return {
    models,
    openRouterModels,
    agents,
    securityAnalyzers,
  };
};

export const useAIConfigOptions = () =>
  useQuery({
    queryKey: ["ai-config-options"],
    queryFn: fetchAiConfigOptions,
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });
