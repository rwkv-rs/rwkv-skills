import { getAdminToken } from "./adminToken";

const SERVER_API_BASE = process.env.SCOREBOARD_API_BASE_URL || "http://127.0.0.1:7860";
const BROWSER_API_BASE =
  process.env.NEXT_PUBLIC_SCOREBOARD_API_BASE_URL || normalizeBasePath(process.env.NEXT_PUBLIC_BASE_PATH);

function apiUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path;
  const normalized = path.startsWith("/") ? path : `/${path}`;
  if (typeof window === "undefined") return joinBaseAndPath(SERVER_API_BASE, normalized);
  return BROWSER_API_BASE ? joinBaseAndPath(BROWSER_API_BASE, normalized) : normalized;
}

function normalizeBasePath(value: string | undefined): string {
  const trimmed = (value || "").trim();
  if (!trimmed) return "";
  const withLeadingSlash = trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
  return withLeadingSlash.replace(/\/+$/, "");
}

function joinBaseAndPath(base: string, path: string): string {
  return `${base.replace(/\/+$/, "")}${path}`;
}

export async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(apiUrl(path), { headers: { Accept: "application/json" }, cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status}: ${await errorText(res)}`);
  return res.json() as Promise<T>;
}

export async function postJson<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(apiUrl(path), {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status}: ${await errorText(res)}`);
  return res.json() as Promise<T>;
}

export async function getJsonAuth<T>(path: string): Promise<T> {
  const res = await fetch(apiUrl(path), { headers: adminHeaders(), cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status}: ${await errorText(res)}`);
  return res.json() as Promise<T>;
}

export async function postJsonAuth<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(apiUrl(path), {
    method: "POST",
    headers: adminHeaders({ "Content-Type": "application/json" }),
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status}: ${await errorText(res)}`);
  return res.json() as Promise<T>;
}

function adminHeaders(extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = { Accept: "application/json", ...extra };
  const token = getAdminToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  return headers;
}

async function errorText(res: Response): Promise<string> {
  try {
    const data = await res.json();
    return typeof data?.detail === "string" ? data.detail : JSON.stringify(data);
  } catch {
    try {
      return await res.text();
    } catch {
      return res.statusText;
    }
  }
}
