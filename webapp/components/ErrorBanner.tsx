import { AlertCircle } from "lucide-react";

interface ErrorBannerProps {
  message: string | null;
}

// Dumb display component — renders whatever friendly string it receives.
// The mapping from raw ApiError → friendly text happens in the caller (page.tsx).
// Raw error details (status, schema fields, stack) go to console.error, never here.
export default function ErrorBanner({ message }: ErrorBannerProps) {
  if (!message) return null;

  return (
    <div
      role="alert"
      className="mt-6 flex items-start gap-3 border border-party-rep/30 bg-party-rep/5 px-4 py-3 text-sm italic text-party-rep"
    >
      <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
      <span>{message}</span>
    </div>
  );
}
