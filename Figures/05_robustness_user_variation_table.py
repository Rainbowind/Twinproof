from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import cm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


OUTPUT = Path(__file__).with_suffix(".pdf")

metrics = ["Forged", "Replay", "Proxy", "Trans.", "Overall"]
posture_variations = ["Hand-held", "In-pocket", "In-bag"]
motion_variations = ["Walk", "Run"]
variations = posture_variations + motion_variations
values = [
    [2.58, 2.66, 2.73, 2.61, 2.86],
    [7.08, 7.22, 7.36, 7.12, 7.58],
    [3.55, 3.63, 3.72, 3.57, 3.89],
    [3.06, 3.15, 3.23, 3.08, 3.38],
    [4.07, 4.17, 4.26, 4.10, 4.43],
]


def draw_centered_text(pdf, text, x, y, width, font_name="Times-Roman", font_size=8):
    pdf.setFont(font_name, font_size)
    text_width = stringWidth(text, font_name, font_size)
    pdf.drawString(x + (width - text_width) / 2, y, text)


def main():
    page_w, page_h = landscape((9.2 * cm, 3.55 * cm))
    pdf = canvas.Canvas(str(OUTPUT), pagesize=(page_w, page_h))

    left = 0.45 * cm
    top = page_h - 0.38 * cm
    table_w = page_w - 0.9 * cm

    metric_w = 1.55 * cm
    group_gap = 0.35 * cm
    col_w = (table_w - metric_w - group_gap) / len(variations)

    row_h = 0.38 * cm
    y_top = top
    y_header1 = y_top - 0.28 * cm
    y_header2 = y_header1 - row_h
    y_midrule = y_header2 - 0.20 * cm
    first_data_y = y_midrule - 0.34 * cm

    x_posture_start = left + metric_w
    x_posture_end = x_posture_start + len(posture_variations) * col_w
    x_motion_start = x_posture_end + group_gap
    x_motion_end = x_motion_start + len(motion_variations) * col_w

    column_starts = [
        x_posture_start,
        x_posture_start + col_w,
        x_posture_start + 2 * col_w,
        x_motion_start,
        x_motion_start + col_w,
    ]

    # Three-line table rules.
    pdf.setStrokeColor(colors.black)
    pdf.setLineWidth(0.75)
    pdf.line(left, y_top, left + table_w, y_top)
    pdf.setLineWidth(0.45)
    pdf.line(x_posture_start, y_header1 - 0.12 * cm, x_posture_end, y_header1 - 0.12 * cm)
    pdf.line(x_motion_start, y_header1 - 0.12 * cm, x_motion_end, y_header1 - 0.12 * cm)
    pdf.line(left, y_midrule, left + table_w, y_midrule)

    # Headers.
    draw_centered_text(
        pdf,
        "Holding Posture",
        x_posture_start,
        y_header1,
        x_posture_end - x_posture_start,
        font_name="Times-Roman",
        font_size=8.2,
    )
    draw_centered_text(
        pdf,
        "Walking Mode",
        x_motion_start,
        y_header1,
        x_motion_end - x_motion_start,
        font_name="Times-Roman",
        font_size=8.2,
    )
    draw_centered_text(pdf, "Metric", left, y_header2, metric_w, font_size=7.6)

    for idx, label in enumerate(variations):
        draw_centered_text(
            pdf,
            label,
            column_starts[idx],
            y_header2,
            col_w,
            font_size=7.4,
        )

    # Data rows.
    for row_idx, metric in enumerate(metrics):
        y = first_data_y - row_idx * row_h
        draw_centered_text(pdf, metric, left, y, metric_w, font_size=7.7)
        for col_idx, value in enumerate(values[row_idx]):
            draw_centered_text(
                pdf,
                f"{value:.2f}",
                column_starts[col_idx],
                y,
                col_w,
                font_size=7.7,
            )

    y_bottom = first_data_y - (len(metrics) - 1) * row_h - 0.18 * cm
    pdf.setLineWidth(0.75)
    pdf.line(left, y_bottom, left + table_w, y_bottom)

    pdf.save()


if __name__ == "__main__":
    main()
