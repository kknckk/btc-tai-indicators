import Link from 'next/link';

export default function LiteraturaLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const papers = [
    { id: 1, title: 'Network Activity & Volatility' },
    { id: 2, title: 'On-Chain Flows & Returns' },
    { id: 3, title: 'ARDL: SOPR & Active Addresses' },
    { id: 4, title: 'Bitcoin Cycles (MVRV & NUPL)' },
    { id: 5, title: 'Prophet & ML Forecasting' },
    { id: 6, title: 'Transformer Volatility' },
    { id: 7, title: 'SVM & Random Forest Direction' },
    { id: 8, title: 'Cohort Analysis (UTXO)' },
    { id: 9, title: 'Market Structure & Exchange Flows' },
    { id: 10, title: 'PCA Early Warning Indicators' },
  ];

  return (
    <div className="flex flex-col md:flex-row min-h-screen">
      <aside className="w-full md:w-64 bg-gray-50 border-r p-4 h-full">
        <h2 className="text-xl font-bold mb-4 text-gray-800">Literatura</h2>
        <ul className="space-y-2">
          {papers.map((p) => (
            <li key={p.id}>
              <Link href={`/literatura/${p.id}`} className="block p-2 rounded hover:bg-blue-100 text-sm text-gray-700 font-medium">
                {p.id}. {p.title}
              </Link>
            </li>
          ))}
        </ul>
      </aside>
      <main className="flex-1 p-6">
        {children}
      </main>
    </div>
  );
}
