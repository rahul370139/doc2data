/**
 * Catch-all proxy for the FastAPI backend.
 *
 * We use a Route Handler instead of `next.config.mjs` rewrites because
 * `output: "standalone"` bakes the rewrites at build time, so runtime
 * environment variables (e.g. API_BASE_URL) are ignored. This handler
 * reads env at request time and streams the response back to the client,
 * which is important for the SSE `/extract/graph/stream` endpoint.
 */

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const BACKEND = () =>
  process.env.API_BASE_URL ||
  process.env.INTERNAL_API_BASE_URL ||
  "http://localhost:8000";

async function proxy(
  req: Request,
  { params }: { params: { slug: string[] } },
): Promise<Response> {
  const backend = BACKEND();
  const path = "/" + (params.slug?.join("/") ?? "");
  const url = new URL(req.url);
  const target = `${backend}${path}${url.search}`;

  // Buffer the full body. Streaming the ReadableStream through undici's
  // fetch is fragile with multipart uploads in the Next.js route-handler
  // runtime (patched fetch + duplex is unreliable).
  const headers = stripHopByHop(req.headers);
  const init: RequestInit & { duplex?: "half" } = {
    method: req.method,
    headers,
    redirect: "manual",
    cache: "no-store",
  };
  if (!["GET", "HEAD"].includes(req.method)) {
    const bytes = await req.arrayBuffer();
    if (bytes.byteLength > 0) {
      init.body = bytes;
    }
  }

  console.log(
    `[proxy] ${req.method} ${path} → ${target} ` +
      `(body=${init.body ? (init.body as ArrayBuffer).byteLength : 0}B)`,
  );

  let upstream: Response;
  try {
    upstream = await fetch(target, init);
  } catch (err: unknown) {
    const e = err as Error & { cause?: Error };
    console.error(`[proxy] fetch failed: ${e.message} cause=${e.cause?.message}`);
    return new Response(
      JSON.stringify({
        error: "backend_unreachable",
        target,
        message: e.message,
        cause: e.cause?.message,
      }),
      { status: 502, headers: { "content-type": "application/json" } },
    );
  }

  console.log(`[proxy] ← ${upstream.status} ${upstream.headers.get("content-type")}`);

  const outHeaders = new Headers(upstream.headers);
  outHeaders.delete("content-encoding");
  outHeaders.delete("transfer-encoding");
  outHeaders.delete("connection");

  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: outHeaders,
  });
}

function stripHopByHop(src: Headers): Headers {
  const h = new Headers();
  const drop = new Set([
    "host",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "accept-encoding",
    // undici (Node fetch) refuses 100-continue; curl adds this automatically
    // for bodies > 1KB and the browser never sends it, so it's safe to drop.
    "expect",
  ]);
  src.forEach((v, k) => {
    if (!drop.has(k.toLowerCase())) h.set(k, v);
  });
  return h;
}

export { proxy as GET, proxy as POST, proxy as PUT, proxy as DELETE, proxy as PATCH };
