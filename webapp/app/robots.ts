import type { MetadataRoute } from "next";

// Served by Next.js at https://electoral.juhi.studio/robots.txt
export default function robots(): MetadataRoute.Robots {
  const allowAll = { allow: "/" };
  return {
    rules: [
      // Search + AI/answer-engine crawlers, explicitly welcomed for citation.
      { userAgent: "Googlebot", ...allowAll },
      { userAgent: "Bingbot", ...allowAll },
      { userAgent: "GPTBot", ...allowAll },
      { userAgent: "OAI-SearchBot", ...allowAll },
      { userAgent: "ChatGPT-User", ...allowAll },
      { userAgent: "ClaudeBot", ...allowAll },
      { userAgent: "Claude-Web", ...allowAll },
      { userAgent: "anthropic-ai", ...allowAll },
      { userAgent: "PerplexityBot", ...allowAll },
      { userAgent: "Perplexity-User", ...allowAll },
      { userAgent: "Google-Extended", ...allowAll },
      { userAgent: "*", ...allowAll },
    ],
    sitemap: "https://electoral.juhi.studio/sitemap.xml",
  };
}
