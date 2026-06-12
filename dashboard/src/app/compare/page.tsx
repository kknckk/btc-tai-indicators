"use client";

import React, { useState, useMemo } from "react";
import { LineChart, Loader2, Info } from "lucide-react";
import Chart, { ChartSeries } from "@/components/Chart";
import { useAvailableIndicators, useIndicators } from "@/hooks/useIndicators";

const COLORS = [
  "#3b82f6", // blue
  "#ef4444", // red
  "#f59e0b", // amber
  "#8b5cf6", // violet
  "#ec4899", // pink
  "#14b8a6", // teal
  "#f97316", // orange
  "#6366f1", // indigo
];

export default function ComparePage() {
  const { data: availableData, isLoading: isAvailableLoading } = useAvailableIndicators();
  const [selectedIndicators, setSelectedIndicators] = useState<string[]>([]);
  const [showPrice, setShowPrice] = useState(true);

  // Zawsze pytamy o wybrane wskaźniki + ewentualnie PriceUSD
  const metricsToFetch = useMemo(() => {
    const metrics = [...selectedIndicators];
    if (showPrice && !metrics.includes("PriceUSD")) {
      metrics.push("PriceUSD");
    }
    return metrics;
  }, [selectedIndicators, showPrice]);

  const { data, isLoading, isError } = useIndicators(metricsToFetch.length > 0 ? metricsToFetch : ["PriceUSD"], "2012-01-01", "2026-12-31");

  const toggleIndicator = (indicator: string) => {
    if (indicator === "PriceUSD") return;
    setSelectedIndicators(prev => 
      prev.includes(indicator) 
        ? prev.filter(i => i !== indicator)
        : [...prev, indicator]
    );
  };

  const series: ChartSeries[] = useMemo(() => {
    if (!data) return [];
    const s: ChartSeries[] = [];
    
    if (showPrice) {
      const priceData = data.find(d => d.indicator === "PriceUSD")?.data || [];
      if (priceData.length > 0) {
        s.push({
          id: "PriceUSD",
          name: "BTC Price (USD)",
          color: "#10b981", // Emerald
          data: priceData,
          axis: "right",
          logarithmic: true,
        });
      }
    }

    selectedIndicators.forEach((indicator, index) => {
      const indData = data.find(d => d.indicator === indicator)?.data || [];
      if (indData.length > 0) {
        s.push({
          id: indicator,
          name: indicator,
          color: COLORS[index % COLORS.length],
          data: indData,
          axis: "left",
          logarithmic: false,
        });
      }
    });

    return s;
  }, [data, showPrice, selectedIndicators]);

  return (
    <div className="min-h-screen bg-slate-950 p-8 font-sans">
      <div className="max-w-7xl mx-auto space-y-8">
        <header className="flex items-center gap-4">
          <div className="p-4 bg-indigo-900/30 text-indigo-400 rounded-2xl border border-indigo-800 shadow-lg">
            <LineChart className="w-8 h-8" />
          </div>
          <div>
            <h1 className="text-4xl font-extrabold text-white">
              Porównanie Wskaźników
            </h1>
            <p className="text-slate-400 mt-2">
              Nałóż na siebie dowolne wskaźniki, aby wyszukać korelacje.
            </p>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Panel boczny */}
          <div className="lg:col-span-1 bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
            <h2 className="text-xl font-bold text-white mb-4">Wskaźniki</h2>
            
            <label className="flex items-center gap-3 p-3 bg-slate-800/50 rounded-xl cursor-pointer hover:bg-slate-800 transition-colors border border-emerald-900/50">
              <input 
                type="checkbox" 
                className="w-5 h-5 accent-emerald-500 rounded bg-slate-700 border-slate-600"
                checked={showPrice}
                onChange={() => setShowPrice(!showPrice)}
              />
              <span className="text-emerald-400 font-semibold">Cena BTC (USD)</span>
            </label>

            <div className="h-[400px] overflow-y-auto space-y-2 pr-2">
              {isAvailableLoading && <div className="text-slate-500 text-sm">Ładowanie wskaźników...</div>}
              {availableData?.indicators.filter(i => i !== "PriceUSD").sort().map(indicator => (
                <label key={indicator} className="flex items-center gap-3 p-2 hover:bg-slate-800 rounded-lg cursor-pointer transition-colors group">
                  <input 
                    type="checkbox" 
                    className="w-4 h-4 accent-indigo-500 rounded bg-slate-700 border-slate-600"
                    checked={selectedIndicators.includes(indicator)}
                    onChange={() => toggleIndicator(indicator)}
                  />
                  <span className="text-slate-300 group-hover:text-white transition-colors">{indicator}</span>
                </label>
              ))}
            </div>
            {selectedIndicators.length > 5 && (
              <div className="p-3 bg-amber-900/20 border border-amber-800/50 rounded-xl flex items-start gap-2">
                <Info className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
                <p className="text-xs text-amber-400/80">
                  Wybranie wielu wskaźników o innej skali z lewej osi Y może sprawić, że wykres stanie się nieczytelny.
                </p>
              </div>
            )}
          </div>

          {/* Wykres */}
          <div className="lg:col-span-3">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl h-[600px] flex flex-col">
              {isLoading ? (
                <div className="flex-1 flex flex-col items-center justify-center gap-4 text-indigo-500">
                  <Loader2 className="w-10 h-10 animate-spin" />
                  <p className="text-slate-400 text-sm">Przetwarzanie danych dla wykresu...</p>
                </div>
              ) : isError ? (
                <div className="flex-1 flex flex-col items-center justify-center text-red-500">
                  Wystąpił błąd podczas ładowania danych z API.
                </div>
              ) : series.length === 0 ? (
                <div className="flex-1 flex flex-col items-center justify-center text-slate-500">
                  Zaznacz wskaźniki z listy po lewej, aby wygenerować wykres.
                </div>
              ) : (
                <Chart series={series} title="Analiza Porównawcza" />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
