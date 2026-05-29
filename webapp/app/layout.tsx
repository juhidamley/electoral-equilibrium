import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Electoral Equilibrium",
  description: "Voter coalition rebalancing after political shocks — CMC SRP 2026",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
