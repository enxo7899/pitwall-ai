export const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;

export const signed = (x: number, digits = 1) => `${x > 0 ? "+" : x < 0 ? "−" : "±"}${Math.abs(x).toFixed(digits)}`;

/** A number with a real minus sign, for interval ends such as −0.8. */
export const num = (x: number, digits = 1) => `${x < 0 ? "−" : ""}${Math.abs(x).toFixed(digits)}`;

export const thousands = (x: number) => x.toLocaleString("en-GB");
