module "k_w5core.dll"
include "\\lbfiler3\apps\Tools\Energiewirtschaft\Kisters\BelVis\bas\KiBasic_Lib_include.bas"
w5login("PFMSTROM", "s1K1uUZ78hcb", "BELVIS.LICHTBLICK.DE")

integer vb_sql
string sql
string path = "c:\temp\memoized\timeseries.csv"

rem debug start
sql = "b.BOOK_PATH_S, b.NAME_BOOK_S, b.SHORTNAME_BOOK_S, b.IDENT_BOOK_L, "
sql = sql + "z.IDENT_VL_L, z.NAME_ZR_S, z.EINHEIT_S, z.SPECIFICATION_L, z.NAME_SPEC_S, z.LASTSAVE_TS "
sql = sql + "FROM V_SD_BOOK b INNER JOIN V_ZR_ALL z on z.IDENT_INST_L = b.IDENT_BOOK_L "
sql = sql + "ORDER BY b.IDENT_BOOK_L"
rem debug end

sql = "b.IDENT_BOOK_L, z.IDENT_VL_L, z.NAME_ZR_S, z.LASTSAVE_TS "
sql = sql + "FROM V_SD_BOOK b INNER JOIN V_ZR_ALL z on z.IDENT_INST_L = b.IDENT_BOOK_L "
vb_sql = _dbselect(sql)
vbgotop(vb_sql)

vbexport(vb_sql, path, ",", 1,-1, 1,-1)

_vbclose(vb_sql)