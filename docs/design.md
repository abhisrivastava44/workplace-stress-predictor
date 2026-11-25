# design.md - UI design system

## Feel
Calm, trustworthy, modern. It is a wellness tool, so no alarming reds everywhere and no clinical look. Generous whitespace, rounded cards, one clear action per screen.

## Colors (CSS variables)
--bg: #F6F8FB
--surface: #FFFFFF
--text: #1F2937
--text-muted: #6B7280
--primary: #4F46E5        (indigo, buttons, links, progress)
--primary-hover: #4338CA
--border: #E5E7EB
Risk levels (always paired with a text label and icon):
--risk-low: #16A34A   (bg tint #DCFCE7)
--risk-medium: #D97706 (bg tint #FEF3C7)
--risk-high: #DC2626   (bg tint #FEE2E2)
SHAP bars: increases stress = #EF4444, reduces stress = #22C55E.
Dark mode: optional. If added, use CSS variables and prefers-color-scheme with --bg #0F172A, --surface #1E293B, --text #E5E7EB.

## Typography
Font: Inter (Google Fonts) with system-ui, sans-serif fallback.
H1 32px/700, H2 24px/600, H3 18px/600, body 16px/400, small 14px, line-height 1.6.

## Spacing and shape
8px spacing scale. Cards: radius 16px, 1px border, soft shadow (0 4px 16px rgba(0,0,0,0.06)). Buttons: radius 12px, height 44px minimum. Max content width 960px, centered.

## Pages and components
1. Home: title, one-line explanation, "Start check-in" button, disclaimer, link to Insights.
2. Questionnaire: 5 steps with a progress bar: (1) About you, (2) Work, (3) Family and money, (4) Lifestyle, (5) Review. Options as radio-card buttons for short lists, dropdown for longer. Back/Next buttons. Next disabled until required fields are answered.
3. Result: large stress badge (Low/Medium/High with icon + text), three probability bars, stress index gauge (0-100), "What is driving your result" horizontal bar chart (top 5 factors, red/green), "Suggestions" cards, button "Try What-If", disclaimer footer.
4. What-If: left column = current answers (read-only for non-modifiable), modifiable fields as selects with a "changed" highlight; right column = before vs after cards with delta stress index (green arrow down is good). Note about model estimates.
5. Insights: stress distribution by job level (stacked bar), top drivers per level, short interpretation text.
States: skeleton loaders while waiting, friendly error card with retry, empty state text. Note that the free backend may take ~30s to wake up; show "Waking up the server..." message after 5s.

## Accessibility
Contrast AA, visible focus ring (2px primary), labels linked to inputs, never color alone, aria-live for results, works at 375px width.

## Tone of copy
Friendly and neutral: "Your estimated stress level", "What may be driving it", "Small changes to explore". Never "diagnosis", "disorder", or alarming language.
