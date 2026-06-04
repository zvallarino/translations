import openpyxl, os

files = [
    "/home/zvallarino/brightdata/translation/urls.xlsx",
    "/home/zvallarino/brightdata/translation/urls_olds/urls_I_thinktherearethe2KIndiaAndPakistain.xlsx",
    "/home/zvallarino/brightdata/translation/urls_olds/urls2.xlsx",
    "/home/zvallarino/brightdata/translation/urls_olds/urls_new.xlsx",
    "/home/zvallarino/brightdata/translation/old_translations/ind_pak_tri_6_2.xlsx",
    "/home/zvallarino/brightdata/translation/old_translations/urls.xlsx",
]

for f in files:
    try:
        wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(min_row=1, max_row=5, max_col=3, values_only=True))
        has_yt = any("youtube" in str(cell).lower() or "youtu.be" in str(cell).lower()
                     for row in rows for cell in row if cell)
        has_tiktok = any("tiktok" in str(cell).lower() for row in rows for cell in row if cell)
        print(os.path.basename(f) + ": youtube=" + str(has_yt) + ", tiktok=" + str(has_tiktok) + ", headers=" + str(rows[0] if rows else None))
        wb.close()
    except Exception as e:
        print(os.path.basename(f) + ": ERROR - " + str(e))
