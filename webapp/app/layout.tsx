// RootLayout — the outermost HTML shell Next.js wraps around EVERY page.
// Whatever page the user visits is rendered into `{children}` below, inside this
// <html>/<body>. It's the place for app-wide setup: the global stylesheet, the
// page <title>/description metadata (the `metadata` export — Next.js injects it
// into <head>), JSON-LD structured data, and base body styling.

import type { Metadata } from "next";
import "./globals.css";

const DESCRIPTION =
  "Electoral Equilibrium is a stochastic-optimization research pipeline (fine-tuned Mistral 7B, CVXPY optimization, Monte Carlo simulation) that finds a party's stable equilibrium voter coalition and measures how a hypothetical political shock shifts its win probability. A Claremont McKenna SRP 2026 project by Juhi Damley, advised by Prof. Gaston Espinosa. Research tool, not a forecast.";

export const metadata: Metadata = {
  metadataBase: new URL("https://electoral.juhi.studio"),
  title: "Electoral Equilibrium — equilibrium voter coalitions and shock-conditional win probability",
  description: DESCRIPTION,
  authors: [{ name: "Juhi Damley", url: "https://juhi.studio" }],
  alternates: { canonical: "/" },
  openGraph: {
    type: "website",
    title: "Electoral Equilibrium — equilibrium voter coalitions and shock-conditional win probability",
    description: DESCRIPTION,
    url: "https://electoral.juhi.studio",
    siteName: "Electoral Equilibrium",
  },
  twitter: {
    card: "summary_large_image",
    title: "Electoral Equilibrium",
    description: DESCRIPTION,
  },
};

const jsonLd = {
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  name: "Electoral Equilibrium",
  url: "https://electoral.juhi.studio",
  description: DESCRIPTION,
  applicationCategory: "Research",
  operatingSystem: "Web",
  programmingLanguage: ["Python", "TypeScript"],
  codeRepository: "https://github.com/juhidamley/electoral-equilibrium",
  author: {
    "@type": "Person",
    name: "Juhi Damley",
    url: "https://juhi.studio",
    affiliation: {
      "@type": "CollegeOrUniversity",
      name: "Claremont McKenna College",
    },
  },
  isAccessibleForFree: true,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-50 text-gray-900 antialiased">
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
        {children}
      </body>
    </html>
  );
}
