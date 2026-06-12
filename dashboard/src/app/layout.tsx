import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "BTC On-Chain Indicators",
  description: "Advanced Bitcoin on-chain analytics dashboard",
};

import Link from 'next/link';
import { Database, LayoutDashboard, LineChart, Activity } from 'lucide-react';

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-slate-950">
        <nav className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-8 h-16 flex items-center gap-8">
            <Link href="/" className="flex items-center gap-2 text-blue-400 hover:text-blue-300 transition-colors">
              <LayoutDashboard className="w-5 h-5" />
              <span className="font-bold tracking-wide">Wskaźniki</span>
            </Link>
            
            <Link href="/compare" className="flex items-center gap-2 text-slate-400 hover:text-emerald-400 transition-colors">
              <LineChart className="w-5 h-5" />
              <span className="font-medium">Porównaj</span>
            </Link>

            <Link href="/correlation" className="flex items-center gap-2 text-slate-400 hover:text-fuchsia-400 transition-colors">
              <Activity className="w-5 h-5" />
              <span className="font-medium">Korelacje</span>
            </Link>

            <Link href="/data" className="flex items-center gap-2 text-slate-400 hover:text-emerald-400 transition-colors">
              <Database className="w-5 h-5" />
              <span className="font-medium">Baza Danych</span>
            </Link>
          </div>
        </nav>
        {children}
      </body>
    </html>
  );
}
