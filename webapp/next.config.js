/** @type {import('next').NextConfig} */
const nextConfig = {
  // Backend API URL — set NEXT_PUBLIC_API_URL in Vercel env vars
  // or .env.local for local dev.
  // Defaults to Modal deployment; override with HPC vLLM endpoint
  // by setting INFERENCE_BACKEND=hpc in the backend .env.
}

module.exports = nextConfig
