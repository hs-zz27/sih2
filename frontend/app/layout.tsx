import './globals.css';
import { IBM_Plex_Sans, JetBrains_Mono, Manrope } from 'next/font/google';

import Nav from './Nav';

/**
 * Three faces, three jobs.
 *
 * IBM Plex Sans carries display type: headlines and the confidence figure.
 * It is a working typeface drawn for an engineering company rather than a
 * fashion serif, which is the register this thing actually operates in — it
 * reads measurements off satellite imagery and reports its own doubt.
 * Emphasis in a headline is carried by colour, not by an italic. Manrope runs
 * the interface. JetBrains Mono takes every id, unit and telemetry reading, so
 * a number that came out of the pipeline always looks different from a number
 * someone wrote in a sentence.
 *
 * They are self-hosted by next/font rather than linked from Google, so the
 * deck renders the same on the offline demo machine as it does online — the
 * same requirement that made MapView probe for a basemap instead of trusting
 * `navigator.onLine`.
 */
const display = IBM_Plex_Sans({
  subsets: ['latin'],
  weight: ['400', '500', '600'],
  variable: '--font-ibm-plex-sans',
  display: 'swap',
});

const ui = Manrope({
  subsets: ['latin'],
  variable: '--font-manrope',
  display: 'swap',
});

const mono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
  display: 'swap',
});

export const metadata = {
  title: 'SatQuery AI',
  description: 'Interactive Vision-Language Assistant for Remote Sensing',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`${display.variable} ${ui.variable} ${mono.variable}`}
    >
      <body>
        {/* Obsidian ground and film grain sit behind every route, fixed, so
            scrolling a long trace does not scroll the sky with it. */}
        <div className="ground" aria-hidden="true" />
        <div className="grain" aria-hidden="true" />

        {/* One navigation for every route. Without it /models and
            /benchmarks - the two pages that carry the measured evidence -
            were reachable only by typing their URLs. */}
        <Nav />
        {children}
      </body>
    </html>
  );
}
