import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Bitcoin ETF Trend Model",
  description:
    "A daily trend model for a spot Bitcoin ETF, gold and T-bills: paper portfolio on real prices, after Indian tax and charges, next to buy & hold.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
