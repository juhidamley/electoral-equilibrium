"use client";

import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function EstimateCount() {
  const [count, setCount] = useState<number | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/api/audit/count`, { credentials: "include" })
      .then((r) => (r.ok ? r.json() : null))
      .then((data: { count: number } | null) => {
        if (data?.count != null) setCount(data.count);
      })
      .catch(() => {});
  }, []);

  if (count === null) return null;
  return <>{count.toLocaleString()} estimates logged</>;
}
