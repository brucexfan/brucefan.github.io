import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import remarkFootnotes from 'remark-footnotes';
import 'katex/dist/katex.min.css';
import './BlogPostContent.css';
import NotFound from './NotFound';

function BlogPostContent() {
  const [content, setContent] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const { slug } = useParams();

  useEffect(() => {
    setIsLoading(true);
    setError(null);
    fetch(`/blogposts/${slug}.md`)
      .then(response => {
        if (!response.ok) {
          throw new Error('Page Not Found');
        }
        return response.text();
      })
      .then(text => {
        if (text.trim().startsWith('<!doctype html>') || text.trim().startsWith('<!DOCTYPE html>')) {
          throw new Error('Page Not Found');
        }
        setContent(text);
        setIsLoading(false);
      })
      .catch(error => {
        console.error('Error loading blog post:', error);
        setError(error);
        setIsLoading(false);
      });
  }, [slug]);
  
  useEffect(() => {
    const pageBackground = document.querySelector('.page-background');
    if (pageBackground) {
      pageBackground.classList.add('blog-post-page');
      return () => {
        pageBackground.classList.remove('blog-post-page');
      };
    }
  }, []);

  if (isLoading) {
    return <div/>
  }

  if (error) {
    return <NotFound />;
  }

  const components = {
    img: ({node, ...props}) => {
      const src = props.src.startsWith('images/') 
        ? `/blogposts/${props.src}` 
        : `/blogposts/images/${props.src}`;
      return <img {...props} src={src} className="blog-image" alt={props.alt || ''} />;
    }
  };

  return (
    <div className="blog-post-wrapper">
      <div className="blog-post-content">
        <ReactMarkdown 
          remarkPlugins={[remarkGfm, remarkMath, [remarkFootnotes, {inlineNotes: true}]]}
          rehypePlugins={[rehypeKatex, rehypeRaw]}
          components={components}
        >
          {content}
        </ReactMarkdown>
       </div>
     </div>
  );
}

export default BlogPostContent;
