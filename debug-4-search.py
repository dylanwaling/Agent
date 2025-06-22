"""
Debug and testing utilities for Step 4: Search Functionality
Run this to test and debug the search and retrieval process
"""
import time
import numpy as np
import lancedb
from sentence_transformers import SentenceTransformer
import pandas as pd

def test_database_connection():
    """Test connection to the existing database"""
    print("💾 Testing Database Connection")
    print("=" * 40)
    
    try:
        db = lancedb.connect("data/lancedb")
        print("✅ Database connection successful")
        
        # Check if docling table exists
        tables = db.table_names()
        print(f"📊 Available tables: {tables}")
        
        if "docling" in tables:
            table = db.open_table("docling")
            print(f"✅ 'docling' table opened successfully")
            print(f"📊 Table contains {len(table)} rows")
            return table
        else:
            print("❌ 'docling' table not found")
            print("💡 Run 3-embedding.py first to create the database")
            return None
            
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None

def test_embedding_model_loading():
    """Test loading the same embedding model used for indexing"""
    print("\n🤖 Testing Embedding Model")
    print("=" * 40)
    
    try:
        model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        print("✅ SentenceTransformer model loaded successfully")
        
        # Test model consistency
        test_query = "What is Docling?"
        embedding1 = model.encode(test_query)
        embedding2 = model.encode(test_query)
        
        # Check if embeddings are consistent
        similarity = np.dot(embedding1, embedding2)
        print(f"🔍 Model consistency check: {similarity:.6f}")
        
        if similarity > 0.99:
            print("✅ Model is consistent")
        else:
            print("⚠️ Model shows inconsistency")
        
        print(f"📊 Embedding dimension: {len(embedding1)}")
        return model
        
    except Exception as e:
        print(f"❌ Embedding model error: {e}")
        return None

def analyze_database_content(table):
    """Analyze the content structure of the database"""
    print("\n📊 Analyzing Database Content")
    print("=" * 40)
    
    try:
        df = table.to_pandas()
        
        # Basic statistics
        print(f"📈 Database Statistics:")
        print(f"  Total records: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
        # Content analysis
        text_lengths = df['text'].apply(len)
        print(f"📝 Text Content:")
        print(f"  Average length: {text_lengths.mean():.0f} characters")
        print(f"  Min length: {text_lengths.min()} characters")
        print(f"  Max length: {text_lengths.max()} characters")
        
        # Filename analysis
        unique_files = df['filename'].unique()
        print(f"📁 Files:")
        for filename in unique_files:
            count = len(df[df['filename'] == filename])
            print(f"  {filename}: {count} chunks")
        
        # Page analysis
        all_pages = []
        for pages in df['page_numbers'].dropna():
            if pages is not None:
                try:
                    if isinstance(pages, list):
                        all_pages.extend(pages)
                    else:
                        all_pages.append(pages)
                except:
                    pass
        
        if all_pages:
            print(f"📄 Page Coverage:")
            print(f"  Page range: {min(all_pages)}-{max(all_pages)}")
            print(f"  Total page references: {len(all_pages)}")
        
        # Title analysis
        titles_with_content = df[df['title'].notna()]
        print(f"📋 Titles: {len(titles_with_content)} chunks have titles")
        
        # Show sample content
        print(f"\n📝 Sample Content:")
        for i, row in df.head(2).iterrows():
            print(f"  Record {i+1}:")
            print(f"    Title: {row['title']}")
            print(f"    Pages: {row['page_numbers']}")
            print(f"    Text: {str(row['text'])[:150]}...")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Content analysis error: {e}")
        return False

def test_basic_search(table, model):
    """Test basic search functionality"""
    print("\n🔍 Testing Basic Search")
    print("=" * 40)
    
    try:
        # Test queries
        test_queries = [
            "What is Docling?",
            "document processing",
            "AI models",
            "PDF conversion"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            
            # Generate query embedding
            start_time = time.time()
            query_embedding = model.encode(query)
            embed_time = time.time() - start_time
            
            # Perform search
            start_time = time.time()
            results = table.search(query_embedding).limit(3)
            search_time = time.time() - start_time
            
            # Convert to pandas for analysis
            df_results = results.to_pandas()
            
            print(f"  ⚡ Embedding time: {embed_time:.3f}s")
            print(f"  ⚡ Search time: {search_time:.3f}s")
            print(f"  📊 Results: {len(df_results)}")
            
            # Show top result
            if len(df_results) > 0:
                top_result = df_results.iloc[0]
                print(f"  🥇 Top result:")
                print(f"    Title: {top_result['title']}")
                print(f"    Text: {str(top_result['text'])[:100]}...")
                # Note: _distance might not be available in all LanceDB versions
                try:
                    print(f"    Distance: {getattr(top_result, '_distance', 'N/A')}")
                except:
                    pass
        
        return True
        
    except Exception as e:
        print(f"❌ Basic search error: {e}")
        return False

def test_search_accuracy(table, model):
    """Test search accuracy with known queries"""
    print("\n🎯 Testing Search Accuracy")
    print("=" * 40)
    
    try:
        # Known queries that should return relevant results
        accuracy_tests = [
            {
                "query": "Docling",
                "expected_keywords": ["docling", "document", "processing"],
                "description": "Should find Docling-related content"
            },
            {
                "query": "AI models artificial intelligence",
                "expected_keywords": ["ai", "model", "intelligence", "learning"],
                "description": "Should find AI/ML related content"
            },
            {
                "query": "PDF document conversion",
                "expected_keywords": ["pdf", "document", "convert", "format"],
                "description": "Should find PDF processing content"
            }
        ]
        
        total_tests = len(accuracy_tests)
        passed_tests = 0
        
        for i, test in enumerate(accuracy_tests):
            print(f"\n📝 Test {i+1}: {test['description']}")
            print(f"   Query: '{test['query']}'")
            
            # Perform search
            query_embedding = model.encode(test['query'])
            results = table.search(query_embedding).limit(3).to_pandas()
            
            if len(results) > 0:
                # Check if results contain expected keywords
                top_result = results.iloc[0]
                result_text = str(top_result['text']).lower()
                result_title = str(top_result['title']).lower()
                combined_text = result_text + " " + result_title
                
                found_keywords = []
                for keyword in test['expected_keywords']:
                    if keyword.lower() in combined_text:
                        found_keywords.append(keyword)
                
                accuracy = len(found_keywords) / len(test['expected_keywords'])
                print(f"   ✅ Keywords found: {found_keywords}")
                print(f"   📊 Accuracy: {accuracy:.1%}")
                
                if accuracy >= 0.5:  # At least 50% of keywords found
                    passed_tests += 1
                    print(f"   ✅ Test passed")
                else:
                    print(f"   ❌ Test failed")
            else:
                print(f"   ❌ No results returned")
        
        overall_accuracy = passed_tests / total_tests
        print(f"\n📊 Overall Accuracy: {passed_tests}/{total_tests} ({overall_accuracy:.1%})")
        
        return overall_accuracy >= 0.7  # 70% pass rate
        
    except Exception as e:
        print(f"❌ Accuracy test error: {e}")
        return False

def test_search_performance(table, model):
    """Test search performance with multiple queries"""
    print("\n⚡ Testing Search Performance")
    print("=" * 40)
    
    try:
        # Performance test queries
        queries = [
            "document processing",
            "machine learning",
            "PDF extraction",
            "text analysis",
            "data conversion"
        ]
        
        total_embedding_time = 0
        total_search_time = 0
        
        print(f"🔄 Running {len(queries)} search queries...")
        
        for i, query in enumerate(queries):
            # Time embedding generation
            start_time = time.time()
            query_embedding = model.encode(query)
            embedding_time = time.time() - start_time
            total_embedding_time += embedding_time
            
            # Time search
            start_time = time.time()
            results = table.search(query_embedding).limit(5)
            df_results = results.to_pandas()
            search_time = time.time() - start_time
            total_search_time += search_time
            
            if i == 0:  # Show details for first query
                print(f"  Query 1 details:")
                print(f"    Embedding: {embedding_time:.3f}s")
                print(f"    Search: {search_time:.3f}s")
                print(f"    Results: {len(df_results)}")
        
        avg_embedding_time = total_embedding_time / len(queries)
        avg_search_time = total_search_time / len(queries)
        
        print(f"📊 Performance Results:")
        print(f"  Average embedding time: {avg_embedding_time:.3f}s")
        print(f"  Average search time: {avg_search_time:.3f}s")
        print(f"  Total time: {total_embedding_time + total_search_time:.3f}s")
        
        # Performance thresholds
        if avg_embedding_time < 0.1 and avg_search_time < 0.1:
            print("✅ Performance: Excellent")
        elif avg_embedding_time < 0.5 and avg_search_time < 0.5:
            print("✅ Performance: Good")
        else:
            print("⚠️ Performance: Could be improved")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        return False

def test_search_result_quality(table, model):
    """Test the quality and relevance of search results"""
    print("\n🏆 Testing Search Result Quality")
    print("=" * 40)
    
    try:
        test_query = "What is Docling and how does it work?"
        print(f"🔍 Test query: '{test_query}'")
        
        # Get search results
        query_embedding = model.encode(test_query)
        results = table.search(query_embedding).limit(5).to_pandas()
        
        print(f"📊 Retrieved {len(results)} results")
        
        # Analyze result quality
        quality_metrics = {
            "results_with_titles": 0,
            "results_with_pages": 0,
            "avg_text_length": 0,
            "unique_sources": set()
        }
        
        for i, row in results.iterrows():
            print(f"\n📄 Result {i+1}:")
            print(f"  Title: {row['title']}")
            print(f"  Source: {row['filename']}")
            print(f"  Pages: {row['page_numbers']}")
            print(f"  Text length: {len(str(row['text']))} chars")
            print(f"  Preview: {str(row['text'])[:200]}...")
            
            # Update quality metrics
            if row['title'] and str(row['title']) != 'None':
                quality_metrics["results_with_titles"] += 1
            
            if row['page_numbers'] is not None:
                quality_metrics["results_with_pages"] += 1
            
            if row['filename']:
                quality_metrics["unique_sources"].add(row['filename'])
            
            quality_metrics["avg_text_length"] += len(str(row['text']))
        
        # Calculate averages
        if len(results) > 0:
            quality_metrics["avg_text_length"] /= len(results)
        
        print(f"\n📊 Quality Metrics:")
        print(f"  Results with titles: {quality_metrics['results_with_titles']}/{len(results)}")
        print(f"  Results with page info: {quality_metrics['results_with_pages']}/{len(results)}")
        print(f"  Average text length: {quality_metrics['avg_text_length']:.0f} chars")
        print(f"  Unique sources: {len(quality_metrics['unique_sources'])}")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Quality test error: {e}")
        return False

def run_full_search_test():
    """Run the complete search test from 4-search.py"""
    print("\n🚀 Running Full Search Test")
    print("=" * 40)
    
    try:
        # Mirror the exact logic from 4-search.py
        uri = "data/lancedb"
        db = lancedb.connect(uri)
        model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        
        table = db.open_table("docling")
        
        query = "what's docling?"
        query_embedding = model.encode(query)
        
        result = table.search(query_embedding).limit(3)
        df_result = result.to_pandas()
        
        print("✅ Search completed successfully")
        print(f"📊 Results:")
        print(df_result)
        
        return True
        
    except Exception as e:
        print(f"❌ Full search test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Step 4: Search Functionality Debug")
    print("=" * 50)
    
    # Run all tests
    table = test_database_connection()
    model = test_embedding_model_loading()
    
    if table and model:
        analyze_database_content(table)
        test_basic_search(table, model)
        test_search_accuracy(table, model)
        test_search_performance(table, model)
        test_search_result_quality(table, model)
        
        # Final comprehensive test
        print("\n" + "="*50)
        run_full_search_test()
    
    print("\n✅ Search debug completed!")
