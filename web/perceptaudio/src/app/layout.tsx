import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "PerceptAudio 2.0 — Rust-Powered Voice Perception Analyzer",
  description:
    "Real-time microphone awareness with a Rust/WebAssembly DSP engine: who is speaking, where they are, and why — YIN pitch tracking, spectral noise suppression, speaker clustering, live distance estimation and model-powered transcription and intent analysis.",
  keywords: ["voice perception", "rust wasm", "audio analyzer", "speaker diarization", "speech to text", "distance estimation", "yin pitch"],
  authors: [{ name: "Z.ai" }],
  icons: {
    icon: "https://z-cdn.chatglm.cn/z-ai/static/logo.svg",
  },
  openGraph: {
    title: "PerceptAudio 2.0 — Rust-Powered Voice Perception Analyzer",
    description: "WHO is speaking, WHERE they are, WHAT, HOW and WHY — a Rust DSP engine in your browser",
    url: "https://chat.z.ai",
    siteName: "Z.ai",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
