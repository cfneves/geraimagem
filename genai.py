import streamlit as st
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

# Configuração da página
st.set_page_config(page_title="IA Generativa", layout="wide")
st.title("Gerador de Imagens com Stable Diffusion")

# Função para gerar imagens
def generate_images(prompt, negative_prompt, num_images_per_prompt,
                    num_inference_steps, height, width, seed, guidance_scale):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_model_or_path = "stabilityai/stable-diffusion-2-1-base"
    
    # Configuração do agendador e pipeline
    scheduler = EulerDiscreteScheduler.from_pretrained(
        pretrained_model_or_path, subfolder="scheduler"
    )
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_or_path, scheduler=scheduler
    ).to(device)

    # Gerador com seed para reprodução consistente
    generator = torch.Generator(device=device).manual_seed(seed)

    # Geração de imagens
    images = pipeline(
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        generator=generator,
        guidance_scale=guidance_scale
    )['images']

    return images

# Adicionando o Menu Expansível com Explicação
with st.expander("ℹ️ Como funciona a geração de imagens?", expanded=False):
    st.markdown(
        """
        Este aplicativo usa **Stable Diffusion**, um modelo de difusão para gerar imagens 
        a partir de descrições textuais. O usuário pode inserir um **prompt** para descrever 
        a imagem desejada e, opcionalmente, um **negative prompt** para evitar certos elementos.
        
        **Configurações disponíveis:**
        - **Prompt:** Descreva a imagem que você quer gerar.
        - **Negative Prompt:** Descreva o que você NÃO quer na imagem.
        - **Número de Passos de Inferência:** Quantos passos o modelo deve seguir para gerar a imagem.
        - **Altura e Largura:** Defina o tamanho da imagem gerada.
        - **Seed:** Use um valor fixo para gerar imagens consistentes em diferentes execuções.
        - **Escala de Orientação:** Define a importância de seguir o prompt fornecido.

        Após clicar no botão **"Gerar Imagem"**, a imagem será gerada e exibida na tela.
        """
    )

# Configurações na barra lateral
with st.sidebar:
    st.header("Configurações da Geração da Imagem")
    prompt = st.text_area("Prompt", "")
    negative_prompt = st.text_area("Negative Prompt", "")
    num_images_per_prompt = st.slider("Número de Imagens", min_value=1, max_value=5, value=1)
    num_inference_steps = st.number_input("Número de Passos de Inferência", min_value=1, max_value=100, value=50)
    height = st.selectbox("Altura da Imagem", [64, 256], index=1)
    width = st.selectbox("Largura da Imagem", [64, 256], index=1)
    seed = st.number_input("Seed", min_value=0, max_value=99999, value=42)
    guidance_scale = st.number_input("Escala de Orientação", min_value=1.0, max_value=20.0, value=7.5)
    generate_button = st.button("Gerar Imagem")

# Lógica para geração de imagem
if generate_button and prompt:
    with st.spinner("Gerando Imagens..."):
        images = generate_images(
            prompt, negative_prompt, num_images_per_prompt,
            num_inference_steps, height, width, seed, guidance_scale
        )

        # Exibir as imagens geradas em colunas
        cols = st.columns(len(images))
        for idx, (col, img) in enumerate(zip(cols, images)):
            with col:
                st.image(img, caption=f"Imagem {idx + 1}", use_column_width=True, output_format='auto')

# --- Rodapé ---
st.write("---")
st.markdown(
    """
    <div style='text-align: center; margin-top: 20px; line-height: 1.2;'>
        <p style='font-size: 16px; font-weight: bold; margin: 0;'>Projeto: IA Generativa com Stable Diffusion</p>
        <p style='font-size: 14px; margin: 5px 0;'>Desenvolvido por:</p>
        <p style='font-size: 20px; color: #4CAF50; font-weight: bold; margin: 0;'>Cláudio Ferreira Neves</p>
        <p style='font-size: 16px; color: #555; margin: 0;'>Especialista em Análise de Dados, RPA e AI</p>
        <p style='font-size: 14px; margin: 10px 0 5px 0;'>Ferramentas utilizadas: Python, Streamlit, Diffusers, Torch</p>
        <p style='font-size: 12px; color: #777; margin: 0;'>© 2024</p>
    </div>
    """,
    unsafe_allow_html=True
)
