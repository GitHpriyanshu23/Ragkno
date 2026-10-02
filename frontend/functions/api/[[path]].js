const SAFE_METHODS_WITHOUT_BODY = new Set(["GET", "HEAD"]);

function backendUrl(requestUrl, backendOrigin, pathParts) {
  const origin = new URL(backendOrigin);
  if (origin.protocol !== "https:" && origin.hostname !== "127.0.0.1" && origin.hostname !== "localhost") {
    throw new Error("BACKEND_ORIGIN must use HTTPS outside local development.");
  }

  const incoming = new URL(requestUrl);
  const path = Array.isArray(pathParts) ? pathParts.join("/") : (pathParts || "");
  origin.pathname = `/${path}`;
  origin.search = incoming.search;
  return origin;
}

export async function onRequest({ request, env, params }) {
  if (!env.BACKEND_ORIGIN) {
    return new Response("BACKEND_ORIGIN is not configured.", { status: 503 });
  }

  let target;
  try {
    target = backendUrl(request.url, env.BACKEND_ORIGIN, params.path);
  } catch (error) {
    return new Response(error.message, { status: 500 });
  }

  const incoming = new URL(request.url);
  const headers = new Headers(request.headers);
  headers.delete("host");
  headers.set("x-forwarded-host", incoming.host);
  headers.set("x-forwarded-proto", incoming.protocol.slice(0, -1));

  const upstreamRequest = new Request(target, {
    method: request.method,
    headers,
    body: SAFE_METHODS_WITHOUT_BODY.has(request.method) ? undefined : request.body,
    redirect: "manual",
  });

  return fetch(upstreamRequest);
}
