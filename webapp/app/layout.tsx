// RootLayout — the outermost HTML shell Next.js wraps around EVERY page.
// Whatever page the user visits is rendered into `{children}` below, inside this
// <html>/<body>. It's the place for app-wide setup: the global stylesheet, the
// page <title>/description metadata (the `metadata` export — Next.js injects it
// into <head>), and base body styling. There's no per-page logic here.

import type { Metadata } from "next";
import "./globals.css";

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
      <body className="bg-gray-50 text-gray-900 antialiased">{children}</body>
    </html>
  );
}
