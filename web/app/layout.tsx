import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BTC Paper Trader",
  description:
    "Autonomous BTC paper trading: LSTM prediction, walk-forward validation, deterministic risk engine.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
