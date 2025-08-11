#!/usr/bin/env python3
"""
Comprehensive Streamlit Dashboard for Bibliometric Analysis
Carbon Stock Modeling Research with Remote Sensing and AI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Import our analyzer
from bibliometric_analysis import BibliometricAnalyzer

# Page configuration
st.set_page_config(
    page_title="Carbon Stock Modeling Research - Bibliometric Analysis",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E8B57;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4682B4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and process data"""
    analyzer = BibliometricAnalyzer('DF_COMBINADO_LIMPO.csv')
    results = analyzer.create_visualizations()
    return analyzer, results

def create_temporal_chart(temporal_df, selected_years):
    """Create temporal evolution chart"""
    filtered_df = temporal_df[temporal_df['Year'].isin(selected_years)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_df['Year'],
        y=filtered_df['Total'],
        mode='lines+markers',
        name='Total Publications',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=filtered_df['Year'],
        y=filtered_df['Brazilian'],
        mode='lines+markers',
        name='Brazilian Studies',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=filtered_df['Year'],
        y=filtered_df['International'],
        mode='lines+markers',
        name='International Studies',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Temporal Evolution of Publications',
        xaxis_title='Year',
        yaxis_title='Number of Publications',
        template='plotly_white',
        height=500
    )
    return fig

def create_ai_techniques_chart(ai_df, top_n):
    """Create AI techniques chart"""
    filtered_df = ai_df.head(top_n)
    
    fig = px.bar(
        filtered_df,
        x='Frequency',
        y='Technique',
        orientation='h',
        title=f'Top {top_n} AI Techniques in Carbon/Biomass Estimation',
        color='Frequency',
        color_continuous_scale='reds'
    )
    fig.update_layout(
        template='plotly_white',
        height=max(400, top_n * 30),
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig

def create_data_types_chart(data_df, top_n):
    """Create data types chart"""
    filtered_df = data_df.head(top_n)
    
    fig = px.bar(
        filtered_df,
        x='Frequency',
        y='Data_Type',
        orientation='h',
        title=f'Top {top_n} Data Types for Carbon/Biomass Estimation',
        color='Frequency',
        color_continuous_scale='greens'
    )
    fig.update_layout(
        template='plotly_white',
        height=max(400, top_n * 30),
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig

def create_trends_chart(trends_df, selected_items, chart_type):
    """Create trends chart"""
    fig = go.Figure()
    colors = px.colors.qualitative.Set3
    
    for i, item in enumerate(selected_items):
        if item in trends_df.columns:
            fig.add_trace(go.Scatter(
                x=trends_df['Year'],
                y=trends_df[item],
                mode='lines+markers',
                name=item,
                line=dict(width=3, color=colors[i % len(colors)]),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title=f'{chart_type} Trends Over Time',
        xaxis_title='Year',
        yaxis_title='Frequency',
        template='plotly_white',
        height=600
    )
    return fig

def create_geographic_chart(geo_df, top_n):
    """Create geographic distribution chart"""
    filtered_df = geo_df.head(top_n)
    
    fig = px.bar(
        filtered_df,
        x='Publications',
        y='Country',
        orientation='h',
        title=f'Top {top_n} Countries by Publication Count',
        color='Publications',
        color_continuous_scale='blues'
    )
    fig.update_layout(
        template='plotly_white',
        height=max(400, top_n * 25),
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig



def main():
    # Load data
    analyzer, results = load_data()
    
    # Main title
    st.markdown('<h1 class="main-header">🌱 Carbon Stock Modeling Research</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">Bibliometric Analysis: Multisensor Modeling with Remote Sensing and AI</h2>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Analysis Filters")
    
    # Year filter
    years = list(range(2015, 2026))
    selected_years = st.sidebar.multiselect(
        "Select Years",
        years,
        default=years,
        help="Filter data by publication year"
    )
    
    # Study type filter
    study_types = st.sidebar.multiselect(
        "Study Type",
        ["Brazilian", "International", "All"],
        default=["All"],
        help="Filter by study origin"
    )
    
    # Top N filter
    top_n = st.sidebar.slider(
        "Number of items to display",
        min_value=5,
        max_value=20,
        value=10,
        help="Select number of top items to display in charts"
    )
    
    # Main content
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview", 
        "👥 Authors & Journals", 
        "🔬 AI Techniques", 
        "📡 Data Types", 
        "🚁 Drone Analysis", 
        "🌍 Geographic Distribution"
    ])
    
    with tab1:
        st.markdown('<h2 class="sub-header">📊 Research Overview</h2>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Studies", f"{len(analyzer.df_filtered):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Brazilian Studies", f"{len(analyzer.brazil_df):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("International Studies", f"{len(analyzer.international_df):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Drone/UAV Studies", f"{results['drone_count']:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Temporal evolution
        st.plotly_chart(
            create_temporal_chart(results['temporal_df'], selected_years),
            use_container_width=True
        )

        st.markdown("<h2 class='sub-header'>☁️ Word Cloud & Key Terms</h2>", unsafe_allow_html=True)

       # Acessa a figura da nuvem de palavras
        wordcloud_fig = results.get('wordcloud_fig')

        if wordcloud_fig:
            # --- INÍCIO DA MUDANÇA: CRIA COLUNAS PARA CENTRALIZAR ---
            col1, col2, col3 = st.columns([0.2, 1, 0.2]) # Colunas laterais menores para empurrar o conteúdo central
            with col2:
                st.pyplot(wordcloud_fig)
            # --- FIM DA MUDANÇA ---
        else:
            st.warning("No keyword data available for the selected period.")

        # Caixa de insights com os termos mais frequentes
        st.markdown('<div class="insight-box" style="margin-top: 2rem;">', unsafe_allow_html=True)
        st.markdown("**🔑 Featured Terms in the Cloud:**")

        # Acessa o DataFrame de palavras
        top_words_df = results.get('top_words_df')

        if top_words_df is not None and not top_words_df.empty:
            # Exibe os 3 principais termos
            for i, row in top_words_df.head(3).iterrows():
                # Usando os novos nomes de coluna 'Term' e 'Frequency'
                st.markdown(f"• **{row['Term'].title()}**: {row['Frequency']} mentions")

            # Expander com a lista completa
            with st.expander("See the list of the 20 most frequent terms"):
                st.dataframe(top_words_df, use_container_width=True)
                
        st.markdown('</div>', unsafe_allow_html=True)
        
    with tab2:
        st.markdown('<h2 class="sub-header">👥 Authors & Journals Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Top Authors")
            fig_authors = px.bar(
                results['top_authors'].head(top_n),
                x='Publications',
                y='Author',
                orientation='h',
                color='Publications',
                color_continuous_scale='viridis'
            )
            fig_authors.update_layout(
                height=max(400, top_n * 25),
                yaxis={'categoryorder': 'total ascending'},
                template='plotly_white'
            )
            st.plotly_chart(fig_authors, use_container_width=True)
            
            # Brazilian authors
            if len(results['top_brazil_authors']) > 0:
                st.markdown("### Top Brazilian Authors")
                fig_brazil_authors = px.bar(
                    results['top_brazil_authors'].head(10),
                    x='Publications',
                    y='Author',
                    orientation='h',
                    color='Publications',
                    color_continuous_scale='oranges'
                )
                fig_brazil_authors.update_layout(
                    height=300,
                    yaxis={'categoryorder': 'total ascending'},
                    template='plotly_white'
                )
                st.plotly_chart(fig_brazil_authors, use_container_width=True)
        
        with col2:
            st.markdown("### Top Journals")
            fig_journals = px.bar(
                results['top_journals'].head(top_n),
                x='Publications',
                y='Journal',
                orientation='h',
                color='Publications',
                color_continuous_scale='plasma'
            )
            fig_journals.update_layout(
                height=max(400, top_n * 25),
                yaxis={'categoryorder': 'total ascending'},
                template='plotly_white'
            )
            st.plotly_chart(fig_journals, use_container_width=True)
        
        # Top keywords table
        st.markdown("### Most Frequent Keywords")
        st.dataframe(
            results['top_keywords'].head(20),
            use_container_width=True,
            hide_index=True
        )
    
# Em streamlit_dashboard.py, SUBSTITUA o conteúdo da aba "AI & Data" por este:

    with tab3:
        st.markdown("<h2 class='sub-header'>🤖 Most Used AI Techniques</h2>", unsafe_allow_html=True)
        
        # --- Gráfico 1: Frequência Geral (já existente) ---
        ai_techniques_df = results.get('ai_techniques_df')
        if ai_techniques_df is not None and not ai_techniques_df.empty:
            fig_ai_freq = px.bar(
                ai_techniques_df,
                x='Count',
                y='Technique',
                orientation='h',
                title='Frequency of AI Techniques in Carbon Stock Studies',
                color='Count',
                color_continuous_scale=px.colors.sequential.Reds
            )
            fig_ai_freq.update_layout(
                template='plotly_white',
                height=600,
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_ai_freq, use_container_width=True)
        else:
            st.warning("No AI technique data available for the selected period.")

        st.markdown("---") # Linha divisória

        # --- Gráfico 2: Tendências ao Longo do Tempo (NOVO) ---
        st.markdown("<h2 class='sub-header'>📈 Trends of Top 7 AI Techniques Over Time</h2>", unsafe_allow_html=True)
        ai_trends_df = results.get('ai_trends_df')
        if ai_trends_df is not None and not ai_trends_df.empty:
            fig_ai_trends = px.line(
                ai_trends_df,
                x='PY',
                y='Count',
                color='AI_Techniques',
                title='Annual Usage of Top 7 AI Techniques',
                markers=True,
                labels={'PY': 'Publication Year', 'Count': 'Number of Publications', 'AI_Techniques': 'AI Technique'}
            )
            fig_ai_trends.update_layout(
                template='plotly_white',
                height=500,
                legend_title_text='Technique'
            )
            st.plotly_chart(fig_ai_trends, use_container_width=True)
        else:
            st.warning("No AI technique trend data available to display.")

        ai_techniques_df = results.get('ai_techniques_df')
        ai_trends_df = results.get('ai_trends_df')

        if ai_techniques_df is not None and not ai_techniques_df.empty:
            # Cria a caixa de insights
            st.markdown('<div class="insight-box" style="margin-top: 2rem;">', unsafe_allow_html=True)
            st.markdown("**🤖 AI Techniques Insights:**")

            # Insight 1: Técnica mais dominante
            top_technique = ai_techniques_df.iloc[0]
            st.markdown(f"• **Dominant Method**: {top_technique['Technique']} is the most applied, featured in {top_technique['Count']} studies.")

            # Insight 2: Segunda técnica mais popular
            if len(ai_techniques_df) > 1:
                second_technique = ai_techniques_df.iloc[1]
                st.markdown(f"• **Key Contender**: {second_technique['Technique']} follows as a popular and robust alternative.")

            # Insight 3: Tendência mais forte no último ano
            if ai_trends_df is not None and not ai_trends_df.empty:
                # Encontra o último ano com dados e a técnica mais usada nesse ano
                latest_year = ai_trends_df['PY'].max()
                latest_trends = ai_trends_df[ai_trends_df['PY'] == latest_year]
                
                if not latest_trends.empty:
                    top_recent_technique = latest_trends.loc[latest_trends['Count'].idxmax()]
                    st.markdown(f"• **Recent Trend**: **{top_recent_technique['AI_Techniques']}** shows strong recent usage, peaking in {int(latest_year)}.")

            st.markdown('</div>', unsafe_allow_html=True )   
    
    with tab4:
        st.markdown('<h2 class="sub-header">📡 Data Types Analysis</h2>', unsafe_allow_html=True)
        
        # Data types frequency
        st.plotly_chart(
            create_data_types_chart(results['data_types_df'], top_n),
            use_container_width=True
        )
        
        # Data types trends
        st.markdown("### Data Types Trends Over Time")
        available_data_types = results['data_types_df']['Data_Type'].tolist()
        selected_data_types = st.multiselect(
            "Select data types to compare trends:",
            available_data_types,
            default=available_data_types[:5]
        )
        
        if selected_data_types:
            fig_data_trends = create_trends_chart(
                results['data_trends_df'], 
                selected_data_types, 
                "Data Types"
            )
            st.plotly_chart(fig_data_trends, use_container_width=True)
        
        # Insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**🔍 Data Types Insights:**")
        top_3_data = results['data_types_df'].head(3)
        for i, row in top_3_data.iterrows():
            st.markdown(f"• **{row['Data_Type']}**: {row['Frequency']} studies")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<h2 class="sub-header">🚁 Drone/UAV Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Drone usage over time
            drone_years = list(range(2015, 2026))
            drone_counts = [results['drone_by_year'].get(year, 0) for year in drone_years]
            
            fig_drone = go.Figure()
            fig_drone.add_trace(go.Scatter(
                x=drone_years,
                y=drone_counts,
                mode='lines+markers',
                name='Drone/UAV Studies',
                line=dict(color='#ff6b6b', width=4),
                marker=dict(size=10),
                fill='tonexty'
            ))
            
            fig_drone.update_layout(
                title='Drone/UAV Usage Over Time',
                xaxis_title='Year',
                yaxis_title='Number of Studies',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig_drone, use_container_width=True)
        
        with col2:
            # Most promising drone data
            fig_drone_data = px.bar(
                results['promising_drone_data'],
                x='Frequency',
                y='Data_Type',
                orientation='h',
                title='Most Promising Drone Data Types',
                color='Frequency',
                color_continuous_scale='oranges'
            )
            fig_drone_data.update_layout(
                height=400,
                yaxis={'categoryorder': 'total ascending'},
                template='plotly_white'
            )
            st.plotly_chart(fig_drone_data, use_container_width=True)
        
        # Drone insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**🔍 Drone/UAV Insights:**")
        st.markdown(f"• {results['drone_count']} studies ({(results['drone_count']/len(analyzer.df_filtered)*100):.1f}%) use drone/UAV technology")
        st.markdown(f"• Peak drone usage: {max(drone_counts)} studies in {drone_years[drone_counts.index(max(drone_counts))]}")
        top_drone_data = results['promising_drone_data'].iloc[0]
        st.markdown(f"• Most promising drone data: **{top_drone_data['Data_Type']}** ({top_drone_data['Frequency']} studies)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab6:
        st.markdown('<h2 class="sub-header">🌍 Geographic Distribution</h2>', unsafe_allow_html=True)
        
        # Geographic distribution
        st.plotly_chart(
            create_geographic_chart(results['geographic_df'], top_n),
            use_container_width=True
        )
        
        # Brazilian vs International pie chart
        comparison_data = {
            'Study Type': ['Brazilian', 'International'],
            'Count': [len(analyzer.brazil_df), len(analyzer.international_df)]
        }
        
        fig_comparison = px.pie(
            comparison_data,
            values='Count',
            names='Study Type',
            title='Brazilian vs International Studies',
            color_discrete_sequence=['#ff7f0e', '#1f77b4']
        )
        fig_comparison.update_layout(height=400)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Geographic insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**🔍 Geographic Insights:**")
        top_3_countries = results['geographic_df'].head(3)
        for i, row in top_3_countries.iterrows():
            st.markdown(f"• **{row['Country']}**: {row['Publications']} publications")
        brazil_rank = results['geographic_df'][results['geographic_df']['Country'] == 'BRAZIL'].index[0] + 1 if 'BRAZIL' in results['geographic_df']['Country'].values else "Not in top 20"
        st.markdown(f"• Brazil ranks #{brazil_rank} globally in carbon stock modeling research")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**📊 Dashboard created for comprehensive bibliometric analysis of carbon stock modeling research (2015-2025)**")
    st.markdown("*Focus: Multisensor modeling with remote sensing and artificial intelligence*")

if __name__ == "__main__":
    main()
