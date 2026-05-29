"use client";

// Main app page — Week 7/8.
// Progressive reveal via SSE EventSource:
//   1. ShockNarrative  (event: deltas)
//   2. CoalitionChart  (event: equilibrium)
//   3. WinGauge        (event: simulation)
//
// TODO (Week 7): wire SSE stream from /api/estimate/stream

export default function HomePage() {
  return (
    <main>
      <h1>Electoral Equilibrium</h1>
      <p>Voter coalition optimizer — Week 7/8 implementation pending.</p>
    </main>
  );
}
