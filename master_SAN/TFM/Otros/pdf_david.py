import fitz
# !  Para el requirements.txt
# PyMuPDF            1.23.8
# PyMuPDFb           1.23.7
# pandas deberia valer cualquier version mas o menos
# openpyxl necesario tambien 

ruta = '/Users/pau/repo_pau/master_SAN/GUIA DEL ALUMNADO EOI.pdf'
def pdf_cleaner(ruta: str) -> tuple[dict, dict, dict, dict]:
    doc = fitz.open(ruta)
    try: 
        num_hojas = doc.page_count
        metadato_pdf = doc.metadata
        metadato_pdf['num_hojas'] = num_hojas
        contenido_hojas = {}
        links_hojas = {}   
        contenido_tablas = {}
        for i in range(doc.page_count):
            page = doc.load_page(i)
            contenido_hojas[f"page_{str(i)}"] = page.get_text().replace('\n',' ')
            links_hojas[f"page_{str(i)}"] = page.get_links()
            tablas = page.find_tables()
            # podriamos pasar el limpiar excel a las tablas
            tablas_hoja = list()
            for j, tabla in enumerate(tablas):
                tabla.to_pandas().to_excel(f"tabla_{j}_pag_{page.number}.xlsx")
                tablas_hoja += [tabla.to_pandas()]
            contenido_tablas[f"page_{str(i)}"] = tablas_hoja
    except Exception as e:
        doc.close()
        # Esto lo cambiamos a un log cuando este todo el proceso montado
        print(f"Excepcion: {e}")
    doc.close()
    return (metadato_pdf, contenido_hojas, links_hojas, contenido_tablas)