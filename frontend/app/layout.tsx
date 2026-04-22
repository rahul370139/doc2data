import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Doc2Data — Agentic Form Extraction",
  description:
    "Agentic LangGraph pipeline for structured extraction from CMS-1500 and other healthcare forms.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="font-sans antialiased">
        <div className="min-h-screen">{children}</div>
      </body>
    </html>
  );
}
