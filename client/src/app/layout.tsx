import type { Metadata } from "next";

import { Providers } from "./providers";
import "../styles.css";

export const metadata: Metadata = {
  title: "RWKV Skills",
  description: "RWKV evaluation dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
