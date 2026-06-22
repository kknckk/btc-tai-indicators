"use client";

import React, { useMemo } from "react";
import { Database, Download, FileSpreadsheet } from "lucide-react";
import { useAvailableIndicators, getIndicatorGroup, GROUP_ORDER } from "@/hooks/useIndicators";

export default function DataPage() {
  const { data, isLoading, isError } = useAvailableIndicators();

  const groupedIndicators = useMemo(() => {
    if (!data?.indicators) return {};
    const groups: Record<string, string[]> = {};
    data.indicators.forEach((indicator) => {
      const group = getIndicatorGroup(indicator);
      if (!groups[group]) groups[group] = [];
      groups[group].push(indicator);
    });
    // Sort within groups
    Object.keys(groups).forEach(g => groups[g].sort());
    return groups;
  }, [data]);

  return (
    <div className="min-h-screen bg-slate-950 p-8 font-sans">
      <div className="max-w-7xl mx-auto space-y-8">
        <header className="flex items-center gap-4">
          <div className="p-4 bg-emerald-900/30 text-emerald-400 rounded-2xl border border-emerald-800 shadow-lg">
            <Database className="w-8 h-8" />
          </div>
          <div>
            <h1 className="text-4xl font-extrabold text-white">
              Baza Danych (CSV)
            </h1>
            <p className="text-slate-400 mt-2">
              Pobierz surowe zbiory danych historycznych wyliczone przez potok BigQuery.
            </p>
          </div>
        </header>

        <main>
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-6">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <FileSpreadsheet className="text-emerald-500" />
                Zbiory danych ({data?.indicators?.length || 0} plików)
              </h2>
            </div>

            {isLoading ? (
              <div className="py-12 flex justify-center text-slate-500">
                Ładowanie plików CSV...
              </div>
            ) : isError ? (
              <div className="py-12 flex justify-center text-red-500">
                Wystąpił błąd podczas ładowania listy plików.
              </div>
            ) : (
              <div className="space-y-10">
                {GROUP_ORDER.map((group) => {
                  const indicators = groupedIndicators[group];
                  if (!indicators || indicators.length === 0) return null;
                  
                  return (
                    <section key={group} className="space-y-4">
                      <h3 className="text-lg font-semibold text-slate-200 border-b border-slate-800 pb-2 flex items-center gap-2">
                        <span className="text-blue-500">#</span>
                        {group}
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                        {indicators.map((indicator, index) => (
                          <a
                            key={indicator}
                            href={`/csv/${indicator}.csv`}
                            download={`${indicator}.csv`}
                            className="flex items-center justify-between p-4 bg-slate-800/50 border border-slate-700 hover:border-emerald-500 hover:bg-slate-800 rounded-xl transition-all group"
                          >
                            <span className="text-slate-300 font-medium group-hover:text-emerald-400 transition-colors flex items-center gap-2">
                              {indicator}.csv
                            </span>
                            <Download className="w-5 h-5 text-slate-500 group-hover:text-emerald-400 transition-colors" />
                          </a>
                        ))}
                      </div>
                    </section>
                  );
                })}
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
