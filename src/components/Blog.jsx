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
