"use client";
import React from "react";
import Link from "next/link";
import { useAvailableIndicators } from "@/hooks/useIndicators";
import { Loader2, Activity } from "lucide-react";

export default function Home() {
  const { data, isLoading, isError } = useAvailableIndicators();

  return (
    <div className="min-h-screen bg-slate-950 p-8 font-sans">
      <div className="max-w-7xl mx-auto space-y-8">
        <header>
          <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
            BTC On-Chain Dashboard
          </h1>
          <p className="text-slate-400 mt-2">
            Real-time & historical Bitcoin network indicators powered by GCP BigQuery
          </p>
        </header>
        
        <main className="space-y-8">
          <section>
            <h2 className="text-2xl font-bold text-white mb-6">Wybierz wskaźnik</h2>
            {isLoading ? (
              <div className="w-full h-48 bg-slate-900 rounded-xl flex items-center justify-center border border-slate-800 shadow-xl">
                <div className="flex flex-col items-center gap-4 text-blue-500">
                  <Loader2 className="w-8 h-8 animate-spin" />
                  <p className="text-slate-400 text-sm">Ładowanie dostępnych wskaźników...</p>
                </div>
              </div>
            ) : isError ? (
              <div className="w-full h-24 bg-slate-900 rounded-xl flex items-center justify-center border border-red-800 text-red-500 shadow-xl">
                Wystąpił błąd podczas pobierania wskaźników z API.
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {data?.indicators.map((indicator) => (
                  <Link 
                    key={indicator} 
                    href={`/indicator/${indicator}`}
                    className="block group"
                  >
                    <div className="bg-slate-900 p-6 rounded-xl border border-slate-800 hover:border-blue-500 transition-colors shadow-lg flex items-center gap-4">
                      <div className="p-3 bg-slate-800 rounded-lg group-hover:bg-blue-900/30 text-blue-400 transition-colors">
                        <Activity className="w-6 h-6" />
                      </div>
                      <span className="text-slate-200 font-semibold group-hover:text-blue-400 transition-colors">
                        {indicator}
                      </span>
                    </div>
                  </Link>
                ))}
              </div>
            )}
          </section>
        </main>
      </div>
    </div>
  );
}
