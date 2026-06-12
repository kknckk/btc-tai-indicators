"use client";

import React, { useState } from "react";
import { Activity, Loader2, ArrowUpRight, ArrowDownRight, Minus } from "lucide-react";
import { useCorrelation } from "@/hooks/useIndicators";

export default function CorrelationPage() {
  const [days, setDays] = useState(90);
  const { data, isLoading, isError } = useCorrelation(days, "PriceUSD");

  const getCorrelationColor = (corr: number) => {
    if (corr > 0.7) return "text-emerald-400 bg-emerald-400/10 border-emerald-400/20";
    if (corr > 0.3) return "text-emerald-300/70 bg-emerald-400/5 border-emerald-400/10";
    if (corr < -0.7) return "text-red-400 bg-red-400/10 border-red-400/20";
    if (corr < -0.3) return "text-red-300/70 bg-red-400/5 border-red-400/10";
    return "text-slate-400 bg-slate-800 border-slate-700";
  };

  const getCorrelationIcon = (corr: number) => {
    if (corr > 0.3) return <ArrowUpRight className="w-5 h-5" />;
    if (corr < -0.3) return <ArrowDownRight className="w-5 h-5" />;
    return <Minus className="w-5 h-5" />;
  };

  return (
    <div className="min-h-screen bg-slate-950 p-8 font-sans">
      <div className="max-w-7xl mx-auto space-y-8">
        <header className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="p-4 bg-fuchsia-900/30 text-fuchsia-400 rounded-2xl border border-fuchsia-800 shadow-lg">
              <Activity className="w-8 h-8" />
            </div>
            <div>
              <h1 className="text-4xl font-extrabold text-white">
                Macierz Korelacji
              </h1>
              <p className="text-slate-400 mt-2">
                Analiza statystyczna korelacji Pearsona wskaźników z ceną Bitcoina.
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2 bg-slate-900 p-2 rounded-xl border border-slate-800">
            {[30, 90, 180, 365].map(d => (
              <button
                key={d}
                onClick={() => setDays(d)}
                className={`px-4 py-2 rounded-lg font-medium transition-all ${
                  days === d 
                  ? 'bg-fuchsia-500/20 text-fuchsia-400 border border-fuchsia-500/50' 
                  : 'text-slate-400 hover:bg-slate-800 hover:text-white border border-transparent'
                }`}
              >
                {d} Dni
              </button>
            ))}
          </div>
        </header>

        <main className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
          {isLoading ? (
            <div className="py-24 flex flex-col items-center justify-center gap-4 text-fuchsia-500">
              <Loader2 className="w-10 h-10 animate-spin" />
              <p className="text-slate-400 text-sm">Obliczanie korelacji z ostatnich {days} dni z wykorzystaniem zasobów API...</p>
            </div>
          ) : isError ? (
            <div className="py-24 flex items-center justify-center text-red-500">
              Wystąpił błąd podczas analizy korelacji. Upewnij się, że backend został poprawnie wdrożony.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {data?.map((item: any) => (
                <div 
                  key={item.indicator}
                  className={`p-4 rounded-xl border flex flex-col gap-3 transition-colors ${getCorrelationColor(item.correlation)}`}
                >
                  <div className="flex justify-between items-start">
                    <span className="font-bold tracking-wide">{item.indicator}</span>
                    {getCorrelationIcon(item.correlation)}
                  </div>
                  
                  <div className="flex items-end gap-2">
                    <span className="text-3xl font-black">
                      {(item.correlation * 100).toFixed(1)}%
                    </span>
                    <span className="text-sm opacity-70 mb-1">
                      (r={(item.correlation).toFixed(2)})
                    </span>
                  </div>
                  
                  <div className="w-full bg-slate-950/50 rounded-full h-1.5 mt-2 overflow-hidden">
                    <div 
                      className="h-full rounded-full bg-current" 
                      style={{ 
                        width: `${Math.abs(item.correlation * 100)}%`,
                        marginLeft: item.correlation < 0 ? 'auto' : '0'
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
