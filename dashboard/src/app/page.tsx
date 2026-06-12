"use client";
import React from "react";
import Chart, { ChartSeries } from "@/components/Chart";
import ApiTester from "@/components/ApiTester";
import { useIndicators } from "@/hooks/useIndicators";
import { Loader2 } from "lucide-react";

export default function Home() {
  // Pobieramy cenę i MVRV Z-Score z naszego API podłączonego do BigQuery
  const { data, isLoading, isError } = useIndicators(["PriceUSD", "MVRV_Z"], "2012-01-01", "2026-12-31");

  let series: ChartSeries[] = [];
  
  if (data) {
    const priceData = data.find(d => d.indicator === "PriceUSD")?.data || [];
    const mvrvData = data.find(d => d.indicator === "MVRV_Z")?.data || [];
    
    series = [
      {
        id: "PriceUSD",
        name: "BTC Price (USD)",
        color: "#10b981", // Szmaragdowy dla ceny
        data: priceData,
        axis: "right",
        logarithmic: true,
      },
      {
        id: "MVRV_Z",
        name: "MVRV Z-Score",
        color: "#3b82f6", // Niebieski dla oscylatora
        data: mvrvData,
        axis: "left",
        logarithmic: false,
      }
    ];
  }

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
          {/* Główny wykres */}
          <section>
            {isLoading ? (
              <div className="w-full h-[500px] bg-slate-900 rounded-xl flex items-center justify-center border border-slate-800 shadow-xl">
                <div className="flex flex-col items-center gap-4 text-blue-500">
                  <Loader2 className="w-10 h-10 animate-spin" />
                  <p className="text-slate-400 text-sm">Pobieranie z BigQuery...</p>
                </div>
              </div>
            ) : isError ? (
              <div className="w-full h-[500px] bg-slate-900 rounded-xl flex items-center justify-center border border-red-800 text-red-500 shadow-xl">
                Wystąpił błąd podczas ładowania danych. Upewnij się, że serwer API działa na porcie 8080.
              </div>
            ) : (
              <Chart series={series} title="Bitcoin: MVRV Z-Score" />
            )}
          </section>

          {/* Tester API */}
          <section>
            <ApiTester />
          </section>
        </main>
      </div>
    </div>
  );
}
