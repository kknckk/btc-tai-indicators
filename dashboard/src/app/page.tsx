"use client";
import React, { useState } from "react";
import Chart from "@/components/Chart";
import ApiTester from "@/components/ApiTester";

// Mockowe dane na potrzeby demonstracji wizualnej
const mockData = [
  { time: "2024-01-01", value: 1.5 },
  { time: "2024-02-01", value: 2.1 },
  { time: "2024-03-01", value: 1.8 },
  { time: "2024-04-01", value: 2.5 },
  { time: "2024-05-01", value: 3.2 },
];

export default function Home() {
  return (
    <div className="min-h-screen bg-slate-950 p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        <header>
          <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
            BTC On-Chain Dashboard
          </h1>
          <p className="text-slate-400 mt-2">
            Real-time & historical Bitcoin network indicators
          </p>
        </header>
        
        <main className="space-y-8">
          {/* Główny wykres */}
          <section>
            <Chart data={mockData} title="MVRV Z-Score (Mock)" />
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
