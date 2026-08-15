import type { MetadataRoute } from "next";

// Served by Next.js at https://electoral.juhi.studio/sitemap.xml
export default function sitemap(): MetadataRoute.Sitemap {
  return [
    {
      url: "https://electoral.juhi.studio",
      lastModified: "2026-07-27",
      changeFrequency: "monthly",
      priority: 1,
    },
  ];
}
