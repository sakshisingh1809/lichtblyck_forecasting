module "k_w5core.dll"
include "\\lbfiler3\apps\Tools\Energiewirtschaft\Kisters\BelVis\bas\KiBasic_Lib_include.bas"
w5login("PFMSTROM", "s1K1uUZ78hcb", "BELVIS.LICHTBLICK.DE")

integer vb_sql
string sql
string path = "c:\temp\memoized\book.csv"

sql = "IDENT_BOOK_L, NAME_BOOK_S, SHORTNAME_BOOK_S, IDENT_PARENT_L, BOOK_PATH_S "
sql = sql + "FROM V_SD_BOOK ORDER BY IDENT_PARENT_L"
vb_sql = _dbselect(sql)
vbgotop(vb_sql)

vbexport(vb_sql, path, ",", 1,-1, 1,-1)

_vbclose(vb_sql)

