import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Go live — Bitcoin ETF Trend Model",
  description: "How to run the ETF trend model with real money from India: broker, LRS, automation, tax, safety.",
};

export default function GoLive() {
  return (
    <main className="min-h-dvh bg-zinc-950 text-zinc-100">
      <header className="border-b border-zinc-900">
        <div className="mx-auto flex max-w-3xl items-center justify-between gap-2 px-4 py-3">
          <span className="text-sm font-semibold tracking-tight">Go live with real money</span>
          <Link href="/" className="text-xs text-zinc-400 hover:text-zinc-200">← dashboard</Link>
        </div>
      </header>

      <article className="mx-auto max-w-3xl space-y-4 px-4 py-5 text-sm leading-relaxed text-zinc-300">
        <p className="rounded-md border border-emerald-900/60 bg-emerald-950/20 px-4 py-3 text-emerald-200">
          The automation already runs every trading day as a paper portfolio on real prices. Going live is a
          configuration change, not new code: add broker keys, flip two switches. Do the steps in order, and
          stay on paper until the broker account matches this dashboard for a few weeks.
        </p>

        <Block title="1. What you need">
          <ul className="list-disc space-y-1.5 pl-5">
            <li><b>A US broker that accepts Indian residents and has a trading API.</b>
              <ul className="mt-1 list-[circle] space-y-1 pl-5">
                <li><b>Alpaca</b> — the broker this code talks to today. Paper accounts are free for anyone.
                  Check when you apply whether they open live accounts for residents of India; if they do,
                  going live is just new keys and one switch.</li>
                <li><b>Interactive Brokers</b> — accepts Indian residents and has a full API, but its gateway
                  needs a small always-on computer rather than GitHub Actions alone, and this code would need
                  an IBKR adapter next to the Alpaca one.</li>
                <li>Indian apps (INDmoney, Vested, Appreciate and similar) are easy to open but offer no public
                  trading API, so they cannot be automated.</li>
              </ul>
            </li>
            <li><b>Money sent abroad under the Liberalised Remittance Scheme (LRS)</b>: up to $250,000 per
              person per financial year, with PAN and Form A2 at your bank. Tax Collected at Source of 20%
              applies above ₹10 lakh a year; it is not a cost — you get it back against your income tax.</li>
            <li><b>LRS forbids leverage, margin trading and short selling.</b> The code enforces the same rules
              on its own: buys are paid from cash only, sells never exceed what is held, and it asks the broker
              to switch margin and shorting off as a second lock.</li>
          </ul>
        </Block>

        <Block title="2. Two ways to run it" id="alerts">
          <ul className="list-disc space-y-1.5 pl-5">
            <li><b>Semi-automatic — works with any broker app, no API needed.</b> The job sends a phone alert
              only when your split should change (about once a week), and you place the 2–4 orders yourself in
              INDmoney, Vested, Interactive Brokers or any other app. Set up: install the free <b>ntfy</b> app,
              subscribe to your private topic (the random name stored in the GitHub secret{" "}
              <Code>NTFY_TOPIC</Code>), and adjust your holdings to the split in each alert, at the next US open.
              Keep the topic name private: anyone who knows it can read and post to it.</li>
            <li><b>Fully automatic — needs a broker with an API</b> (steps below). Orders are placed for you
              every trading day, with the safety rules in section 4.</li>
          </ul>
        </Block>

        <Block title="3. Switching on full automation">
          <ol className="list-decimal space-y-1.5 pl-5">
            <li><b>Paper broker first.</b> Open a free Alpaca paper account and create API keys. In the GitHub
              repository add them as secrets <Code>ALPACA_KEY_ID</Code> and <Code>ALPACA_SECRET_KEY</Code>, then
              add the variable <Code>SEND_ORDERS</Code> = <Code>on</Code>. From the next trading day the job places
              paper orders after the open.</li>
            <li><b>Run on paper for 1–3 months.</b> Compare the broker account with this dashboard. They will not
              match to the cent (orders fill a little after the open), but the weights should.</li>
            <li><b>Make the repository private.</b> Actions logs of public repositories are public. The job prints
              no balances, but real money should not run in public. Private repositories get 2,000 free Actions
              minutes a month; this uses about 60.</li>
            <li><b>Open and fund the live account</b> (KYC, LRS remittance, convert to USD).</li>
            <li><b>Swap in live keys and set <Code>ALPACA_LIVE</Code> = <Code>yes</Code>.</b> Start small, and set
              <Code>MAX_ORDER_USD</Code> to cap the size of any single order.</li>
            <li><b>Kill switch:</b> set <Code>SEND_ORDERS</Code> = <Code>off</Code> (GitHub → Settings → Secrets
              and variables → Actions → Variables). It takes effect at the next run, with no code change.</li>
          </ol>
        </Block>

        <Block title="4. What runs every day">
          <ul className="list-disc space-y-1.5 pl-5">
            <li><b>Decide — 22:30 UTC (4:00 am IST), Monday to Friday.</b> Fetch the day&rsquo;s real closes, refuse
              stale, missing or absurd data, record the decision once (it can never be edited afterwards), and
              update the paper portfolio.</li>
            <li><b>Execute — 14:45 UTC (8:15 pm IST), Monday to Friday</b>, when the US market is open in both
              summer and winter time. Only if keys are set and the switch is on: sell first, wait for the fills,
              then buy with cash only.</li>
            <li><b>Alert</b> — right after the decision, if the split moved enough to be worth a trade, a phone
              alert goes to your ntfy topic.</li>
            <li><b>No duplicate orders:</b> every order carries an ID built from the decision date, and the broker
              rejects a second copy. Blocked or inactive account: nothing is sent.</li>
            <li><b>Monitoring:</b> the dashboard shows the last run; GitHub emails you when a run fails.</li>
            <li><b>US holidays:</b> no new closing price means nothing new to decide or trade.</li>
          </ul>
        </Block>

        <Block title="5. Costs with real money">
          <ul className="list-disc space-y-1.5 pl-5">
            <li>Forex markup when sending and bringing back money: 0.5–2% each way depending on the bank, plus
              ₹500–1,500 + GST per remittance. The results assume 1.5% each way.</li>
            <li>Commission: $0 at Alpaca for US stocks and ETFs; about $0.35–1 per order at Interactive Brokers.
              The model trades about 50 times a year.</li>
            <li>Fund fees, already inside the ETF prices: IBIT 0.25% a year, GLD 0.40% (GLDM holds the same gold
              for 0.10%), SGOV 0.09%.</li>
          </ul>
        </Block>

        <Block title="6. Tax in India — confirm with a Chartered Accountant">
          <ul className="list-disc space-y-1.5 pl-5">
            <li>Gains on US-listed ETFs are capital gains on foreign securities — not the 30% crypto (VDA) tax.</li>
            <li>Held under 24 months: taxed at your slab rate. The results assume the top slab, 31.2%. Held 24
              months or more: 12.5% plus cess.</li>
            <li>Losses offset gains in the same year, and unused losses carry forward for 8 years if you file
              your return on time.</li>
            <li>Report the account every year in Schedule FA (foreign assets) of ITR-2 or ITR-3.</li>
            <li>Dividends (from SGOV) may have 25% US tax withheld; claim credit in India with Form 67.</li>
            <li>The trade list on the dashboard gives dates and prices your CA needs (oldest shares sold first).</li>
            <li>US estate tax can apply to non-US persons holding more than $60,000 of US assets. Take advice
              before investing large amounts.</li>
          </ul>
        </Block>

        <Block title="7. Risks">
          <ul className="list-disc space-y-1.5 pl-5">
            <li>The model can lose money. On IBIT&rsquo;s real prices since 2024 its worst fall was about 15% after
              tax, against 43% for holding IBIT; Bitcoin itself has fallen 75% in the past.</li>
            <li>Past results do not promise future ones. The real record is under three years long.</li>
            <li>Prices come from an unofficial free source. If it breaks or returns something odd, the job stops
              instead of trading on bad data — check the heartbeat.</li>
          </ul>
        </Block>
      </article>
    </main>
  );
}

function Block({ title, id, children }: { title: string; id?: string; children: React.ReactNode }) {
  return (
    <section id={id} className="scroll-mt-4 rounded-lg border border-zinc-800 bg-zinc-900/40 p-4">
      <h2 className="mb-2 text-sm font-medium text-zinc-100">{title}</h2>
      {children}
    </section>
  );
}

function Code({ children }: { children: React.ReactNode }) {
  return <code className="rounded bg-zinc-800 px-1 py-0.5 text-[12px] text-zinc-200">{children}</code>;
}
