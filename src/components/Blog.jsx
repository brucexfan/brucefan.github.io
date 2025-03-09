import React from 'react';
import { Link } from 'react-router-dom';
import './Blog.css';

function BlogPost({ title, description, date, readingTime }) {
  const slug = title.toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]+/g, '');

  return (
    <Link to={`/posts/${slug}`} className="blog-post-link">
      <div className="blog-post">
        <h3 className="blog-title">{title}</h3>
        <p className="blog-description">{description}</p>
        <div className="blog-meta">
          <span className="blog-date">{date}</span>
          <span className="blog-reading-time">{readingTime} min read</span>
        </div>
      </div>
    </Link>
  );
}

function Blog() {
  const blogPosts = [
    {
      title: "Learning to Learn Across Domains",
      description: (
        <span>
          Investigating feature reuse vs. representation change across large domain gaps in Model-Agnostic Meta-Learning (MAML).
          This was my final project for 6.7960 (Deep Learning) with&nbsp;
          <a
            href="https://janellecai.github.io/"
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()} // Prevent link propagation
          >
            Janelle Cai
          </a>
          .
        </span>
      ),
      date: "2024-12-10",
      readingTime: 10
    },
    {
      title: "Creating and Defeating Hexhunt",
      description: "Optimizing a word game: Explaining solutions to my HackMIT admissions puzzle.",
      date: "2024-08-17",
      readingTime: 11
    },
    {
      title: "Hello World",
      description: "My first blog post! On writing and originality.",
      date: "2024-08-01",
      readingTime: 3
    }
  ];

  return (
    <div className="blog">
      <h2>Blog</h2>
      <div className="blog-list">
        {blogPosts.map((post, index) => (
          <BlogPost key={index} {...post} />
        ))}
      </div>
    </div>
  );
}

export default Blog;
