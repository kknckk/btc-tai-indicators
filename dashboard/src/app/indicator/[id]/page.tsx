"use client";

import React, { use } from "react";
import Link from "next/link";
import { ArrowLeft, Loader2 } from "lucide-react";
import Chart, { ChartSeries } from "@/components/Chart";
import ApiTester from "@/components/ApiTester";
import { useIndicators } from "@/hooks/useIndicators";

interface PageProps {
  params: Promise<{
    id: string;
  }>;
}

export default function IndicatorPage({ params }: PageProps) {
  // Rozpakowanie parametrów używając hooka `use` z React (wymóg w Next.js 15+)
  const resolvedParams = use(params);
  const indicatorId = resolvedParams.id;

  // Pobieranie ceny (zawsze jako punkt odniesienia) i wybranego wskaźnika
  // Uwaga: Jeśli wybrany wskaźnik to PriceUSD, wystarczy pobrać go raz.
  const metricsToFetch = indicatorId === "PriceUSD" ? ["PriceUSD"] : ["PriceUSD", indicatorId];
  
  const { data, isLoading, isError } = useIndicators(metricsToFetch, "2012-01-01", "2026-12-31");

  let series: ChartSeries[] = [];
  
  if (data) {
    const priceData = data.find(d => d.indicator === "PriceUSD")?.data || [];
    
    // Konfiguracja linii ceny bazowej
    series.push({
      id: "PriceUSD",
      name: "BTC Price (USD)",
      color: "#10b981", // Szmaragdowy
      data: priceData,
      axis: "right",
      logarithmic: true,
    });

    if (indicatorId !== "PriceUSD") {
      const indicatorData = data.find(d => d.indicator === indicatorId)?.data || [];
      series.push({
        id: indicatorId,
        name: `Bitcoin: ${indicatorId}`,
        color: "#3b82f6", // Niebieski dla wskaźnika
        data: indicatorData,
        axis: "left",
        logarithmic: false,
      });
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 p-8 font-sans">
      <div className="max-w-7xl mx-auto space-y-8">
        <header className="flex items-center gap-6">
          <Link 
            href="/"
            className="p-3 bg-slate-900 border border-slate-800 rounded-xl text-slate-400 hover:text-white hover:border-blue-500 transition-colors shadow-lg"
          >
            <ArrowLeft className="w-6 h-6" />
          </Link>
          <div>
            <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
              {indicatorId}
            </h1>
            <p className="text-slate-400 mt-2">
              Analiza wskaźnika na tle historycznej ceny Bitcoina
            </p>
          </div>
        </header>

        <main className="space-y-8">
          <section>
            {isLoading ? (
              <div className="w-full h-[500px] bg-slate-900 rounded-xl flex items-center justify-center border border-slate-800 shadow-xl">
                <div className="flex flex-col items-center gap-4 text-blue-500">
                  <Loader2 className="w-10 h-10 animate-spin" />
                  <p className="text-slate-400 text-sm">Pobieranie danych o {indicatorId} z BigQuery...</p>
                </div>
              </div>
            ) : isError ? (
              <div className="w-full h-[500px] bg-slate-900 rounded-xl flex items-center justify-center border border-red-800 text-red-500 shadow-xl">
                Wystąpił błąd podczas ładowania danych.
              </div>
            ) : (
              <Chart series={series} title={`Bitcoin: ${indicatorId}`} />
            )}
          </section>

          <section>
            <ApiTester initialMetric={indicatorId} />
          </section>
        </main>
      </div>
    </div>
  );
}
