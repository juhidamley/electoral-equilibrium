import type { Party } from "@/lib/types";
import { oppositeParty, partyName } from "@/lib/editorial";

interface Props {
  winProbability: number | null;
  modeledParty: Party;
  loading?: boolean;
}

const ARC_LENGTH = 157;

function probabilityRead(probability: number, modeledParty: Party): string {
  if (probability >= 0.6) return `The ${partyName(modeledParty)}' edge holds.`;
  if (probability <= 0.4) return `The ${partyName(oppositeParty(modeledParty))}' edge holds.`;
  return "The race remains close.";
}

export default function EditorialWinDial({ winProbability, modeledParty, loading }: Props) {
  const pending = loading || winProbability === null;
  const probability = winProbability ?? 0;
  const percent = Math.round(probability * 100);
  const leader = probability >= 0.5 ? modeledParty : oppositeParty(modeledParty);
  const color = leader === "democrat" ? "var(--party-dem)" : "var(--party-rep)";

  return (
    <div className="flex flex-col items-center text-center">
      <svg
        viewBox="0 0 120 70"
        className="w-[120px]"
        role="img"
        aria-label={pending ? "Win odds loading" : `${partyName(modeledParty)} win odds: ${percent} percent`}
      >
        <path d="M 10 60 A 50 50 0 0 1 110 60" fill="none" stroke="var(--dial-track)" strokeWidth="7" />
        {!pending && (
          <path
            d="M 10 60 A 50 50 0 0 1 110 60"
            fill="none"
            stroke={color}
            strokeWidth="7"
            strokeDasharray={`${(ARC_LENGTH * probability).toFixed(1)} ${ARC_LENGTH}`}
          />
        )}
        <text x="60" y="55" textAnchor="middle" fill="var(--ink)" fontFamily="Georgia, 'Times New Roman', serif" fontSize="24">
          {pending ? "—" : `${percent}%`}
        </text>
      </svg>
      <p className="editorial-kicker mt-1 text-[10px] text-muted">
        {partyName(modeledParty, true)} win odds
      </p>
      {!pending && <p className="mt-1 text-xs italic text-body">{probabilityRead(probability, modeledParty)}</p>}
    </div>
  );
}
