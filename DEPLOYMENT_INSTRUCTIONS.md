# Instrukcje i Kontekst Wdrożenia (Deployment Context)

Ten dokument zawiera najważniejsze informacje na temat infrastruktury serwowanej w Google Cloud (Cloud Run) oraz procedur deploymentu, aby w przyszłości uniknąć nieporozumień z adresami URL i środowiskami.

## Aplikacja Frontendowa (Dashboard)

Aplikacja Dashboard w Next.js jest ustrukturyzowana i wdrażana na infrastrukturze **Google Cloud Run**.

1. **Projekt GCP**: \`btc-ind\`
2. **Region**: \`europe-central2\` (Warszawa)
3. **Nazwa usługi (Service Name)**: \`btc-dashboard\`

### Konwencje URL Google Cloud Run
Należy pamiętać, że Google Cloud Run **przypisuje każdej usłudze dwa identyczne aliasy domenowe**, które serwują dokładnie tę samą, najnowszą zdeployowaną rewizję:

- **Adres ze skrótem (stary system hashowania GCP)**:
  👉 \`https://btc-dashboard-ml3nc7w4ja-lm.a.run.app\`
- **Adres z numerem projektu (nowy system GCP)**:
  👉 \`https://btc-dashboard-203925818774.europe-central2.run.app\`

**WAŻNE:** Jakikolwiek deployment na usługę \`btc-dashboard\` w tym projekcie i regionie, **automatycznie zaktualizuje oba adresy URL**. Testerzy korzystający z linku zawierającego hash (\`ml3nc7w4ja\`) mają pełny dostęp do nowych funkcji po każdym deploymencie.

### Komenda do wdrożenia Dashboardu
Podczas deployu dashboardu należy zawsze operować z katalogu \`dashboard\` i zbudować aplikację. Flaga \`--source .\` wykorzystuje Buildpacki Cloud Run:

\`\`\`bash
cd dashboard
npm run build
gcloud run deploy btc-dashboard --source . --region europe-central2 --allow-unauthenticated --project btc-ind
\`\`\`

---

## Aplikacja Backendowa (API / Python)

Aplikacja FastAPI dostarczająca dane jest zdeployowana pod nazwą \`btc-api\`.

1. **Projekt GCP**: \`btc-ind\`
2. **Region**: \`europe-central2\`
3. **Nazwa usługi**: \`btc-api\`

### Komenda do wdrożenia API
API w Pythonie korzysta z dostarczonego w katalogu \`api\` pliku Dockerfile.

\`\`\`bash
cd api
gcloud run deploy btc-api --source . --region europe-central2 --allow-unauthenticated --project btc-ind
\`\`\`

---

## Baza Danych (GCP BigQuery)
Wszystkie dane CSV (56 wskaźników) ładowane są do BigQuery.
- Dataset: \`btc_indicators\`
- Tabela: \`btc-ind.btc_indicators.btc_indicators\`

Zapisy do bazy obsługiwane są za pomocą konta ADC (Application Default Credentials). W przypadku problemów z autoryzacją środowiska lokalnego z BQ, konieczne jest podanie klucza autoryzacyjnego np. przez użycie Bearer Tokenu.

## Podsumowanie Architektury
- Zewnętrzne API chronione są limitami darmowych tierów, pobierane za pomocą Python Fetcherów do folderu \`data_ingestion/csv\`.
- \`merge_data.py\` odpowiada za scalanie danych surowych w czasie.
- \`compute_derived_metrics.py\` generuje ostateczne tabele, mockuje braki skomplikowanych wskaźników (SplyAct) i wylicza metryki zrewersowane algebrą.
- Interfejs Web w React znajduje się w folderze \`dashboard\` i operuje bezpośrednio na połączeniu z bazą GCP i dedykowanym backendem.
