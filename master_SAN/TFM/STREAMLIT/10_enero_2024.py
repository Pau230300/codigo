import streamlit st

# Crear un contenedor personalizado con fondo blanco
st.markdown(
    """
    <style>
    .stApp {
        background-color: rgb(254, 254, 254); /* Blanco como color de fondo */
    }

    /* Puedes agregar más personalizaciones de CSS aquí usando los colores extraídos */
    /* Color principal: rgb(234, 3, 4) - Rojo */
    /* Color secundario: rgb(240, 128, 129) - Rojo claro */

    /* Por ejemplo, para el título */
    h1 {
        color: rgb(234, 3, 4); /* Rojo para el título principal */
    }

    /* Para otros elementos, como botones o texto, puedes usar: */
    /* .stButton > button {
        background-color: rgb(240, 128, 129); /* Rojo claro para botones */
    /* }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create centered main title with an image instead of emoji
st.markdown("""
    <h1 style="text-align: center;">
        <img src="https://1000logos.net/wp-content/uploads/2017/09/Santander-Logo.png" alt="Logo" style="height: 300px;"/>
    </h1>
    """, unsafe_allow_html=True)

# Create centered main title
st.title('Chat Consultivo Santander')
st.subheader('''¡Hola! Soy un chatbot consultivo del banco Santander basado en Mistral 7b y, por ahora, puedo responder a cualquier pregunta relacionada con el FAQ.
¡Encantando de ayudarte!
''')
# Create a text input box for the user
prompt = st.text_input('Introduzca su pregunta aquí')

# If the user hits enter
if prompt:
  response = Mistral.llm_q(prompt)
  # ...and write it out to the screen
  st.write(response)

  # Display raw response object

  # Display source text
  with st.expander('Contexto obtenido'):
    #rag_context_json = from_context_raw_to_json(query(prompt=prompt, k=3))
    st.write('Contexto obtenido')

