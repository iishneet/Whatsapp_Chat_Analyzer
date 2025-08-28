import streamlit as st
import preprocessor,helper
import matplotlib.pyplot as plt
import seaborn as sns

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall")

    selected_user = st.sidebar.selectbox("Show analysis",user_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user,df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df)
        fig,ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Word Cloud")
        df_wc = helper.create_wordcloud(selected_user,df)
        fig,ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        most_common_df = helper.most_common_words(selected_user,df)

        fig,ax = plt.subplots()

        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most Common words')
        st.pyplot(fig)

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df['count'].head(), labels=emoji_df['emoji'].head(), autopct="%0.2f%%")
            st.pyplot(fig)

        # Sentiment Analysis
        sentiment_df = helper.sentiment_analysis(df, selected_user)
        # Sentiment Analysis Section
        st.title("Sentiment Analysis")

        # Overall sentiment distribution
        sentiment_counts = sentiment_df['sentiment_label'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=['green', 'red', 'blue'])
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig1)

        # Sentiment over time
        st.title("Sentiment Over Time")
        sentiment_over_time = sentiment_df.groupby('only_date')['sentiment'].mean().reset_index()
        fig2, ax2 = plt.subplots()
        ax2.plot(sentiment_over_time['only_date'], sentiment_over_time['sentiment'], color='purple')
        plt.xticks(rotation='vertical')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Average Sentiment')
        st.pyplot(fig2)

        # Display most positive and negative messages
        st.title("Most Positive and Negative Messages")

        # Most positive message
        most_positive = sentiment_df.loc[sentiment_df['sentiment'].idxmax()]
        st.subheader("Most Positive Message")
        st.write(f"User: {most_positive['user']}")
        st.write(f"Message: {most_positive['message']}")
        st.write(f"Sentiment Score: {most_positive['sentiment']:.2f}")

        # Most negative message
        most_negative = sentiment_df.loc[sentiment_df['sentiment'].idxmin()]
        st.subheader("Most Negative Message")
        st.write(f"User: {most_negative['user']}")
        st.write(f"Message: {most_negative['message']}")
        st.write(f"Sentiment Score: {most_negative['sentiment']:.2f}")

        # Sentiment summary
        st.title("Sentiment Summary")
        sentiment_summary = helper.sentiment_summary(selected_user, sentiment_df)
        st.dataframe(sentiment_summary)

        # Pie chart for sentiment summary
        fig, ax = plt.subplots()
        ax.pie(sentiment_summary['Count'], labels=sentiment_summary['Sentiment'], autopct="%0.2f%%")
        st.pyplot(fig)















