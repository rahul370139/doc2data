/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "standalone",
  // Backend proxying is handled by `app/api/backend/[...slug]/route.ts`
  // (a runtime Route Handler) so that API_BASE_URL is read at request
  // time rather than baked into the build manifest.
};

export default nextConfig;
