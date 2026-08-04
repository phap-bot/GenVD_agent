import type { Metadata } from "next";
import VideoDubbingStudio from "@/components/video-dubbing-studio";

export const metadata: Metadata = {
  title: "Video Clone - AI Studio",
  description:
    "Studio lá»“ng tiáº¿ng video AI vá»›i quy trÃ¬nh nháº­n dáº¡ng giá»ng nÃ³i, dá»‹ch ká»‹ch báº£n, chá»‰nh sá»­a timeline vÃ  render video hoÃ n chá»‰nh.",
  alternates: {
    canonical: "/",
  },
  openGraph: {
    title: "Video Clone - AI Studio",
    description:
      "CÃ´ng cá»¥ lá»“ng tiáº¿ng video AI cÃ³ quy trÃ¬nh human-in-the-loop Ä‘á»ƒ kiá»ƒm tra ká»‹ch báº£n trÆ°á»›c khi xuáº¥t báº£n.",
    type: "website",
  },
};

const structuredData = {
  "@context": "https://schema.org",
  "@type": "WebApplication",
  name: "Video Clone",
  applicationCategory: "MultimediaApplication",
  operatingSystem: "Windows, macOS, Linux",
  description:
    "AI video dubbing studio for speech recognition, script translation, voice generation, subtitle editing, and final video rendering.",
  featureList: [
    "Video upload",
    "Speech recognition with timestamped script",
    "Editable dubbing timeline",
    "AI voice generation",
    "Subtitle burn-in and final video render",
  ],
};

export default function Page() {
  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }}
      />
      <VideoDubbingStudio />
    </>
  );
}
