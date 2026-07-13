import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Video Dubbing Studio",
  description: "Client-side video dubbing workflow for a local FastAPI backend.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

