// WinGauge — win probability semicircle gauge + 90% CI strip.
// Stays in skeleton/loading state until simulation is non-null.
// Populates on SSE event: simulation.
// TODO: implement gauge rendering

import type { SimulationData } from "@/lib/types";

interface WinGaugeProps {
  simulation: SimulationData | null;
  loading: boolean;
}

export default function WinGauge(_props: WinGaugeProps) {
  return null;
}
