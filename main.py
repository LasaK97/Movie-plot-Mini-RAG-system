import sys
import argparse
from pathlib import Path

from src.config import config, print_config
from src.rag_pipeline import RAGPipeline
from src.utils import print_results, save_json_output

def interactive_mode(rag: RAGPipeline):
    """interactive query mode"""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE".center(70))
    print("=" * 70)
    print("Type your questions about movies (or 'quit' to exit)")
    print("=" * 70 + "\n")

    while True:
        try:
            #get user query
            query = input("Your question: ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", 'q']:
                print('\n Goodbye')
                break

            #process query
            response = rag.query(query, verbose=False)

            print_results(response)
        except KeyboardInterrupt:
            print("\nGoodbye")
            break
        except Exception as e:
            print(f"ERROR: str{e}")

def single_query_mode(rag: RAGPipeline, query: str, output_format: str = 'pretty', output_file:str = None):
    """single query mode"""
    if config.system.verbose:
        print(f"\nProcessing query: {query}")

        # response
    response = rag.query(query, verbose=config.system.verbose)

    # print
    if output_format == 'json':
        print_results(response)
    else:
        print_results(response)

    # save
    if output_file:
        output_path = Path(output_file)
        save_json_output(response, output_path)
        print(f"\nResponse saved to: {output_path}")

def run_examples(rag: RAGPipeline, output_format: str = 'pretty'):
    """run examples"""
    examples_file = Path("examples/sample_queries.txt")
    if not examples_file.exists():
        print(f"Examples file not found: {examples_file}")
        print("Using default example queries instead...\n")

        example_queries = [
            "Which movies feature artificial intelligence?",
            "What horror movies involve haunted houses?",
            "Tell me about movies with time travel plots"
        ]
    else:
        with open(examples_file, 'r') as f:
            example_queries = [
                line.strip()
                for line in f
                if line.strip() and not line.startswith('#')
            ]

    print("\n" + "=" * 70)
    print("RUNNING EXAMPLE QUERIES".center(70))
    print("=" * 70)

    for i, query in enumerate(example_queries, 1):
        print(f"\n[Query {i}/{len(example_queries)}]")

        response = rag.query(query, verbose=False)

        if output_format == 'json':
            print_results(response)
        else:
            print_results(response)

        if i < len(example_queries):
            input("\nPress Enter for next query...")

    print("\n" + "=" * 70)
    print("ALL EXAMPLE QUERIES COMPLETED")
    print("=" * 70)

def main():
    """main function"""
    parser = argparse.ArgumentParser(
        description="Movie Plots Mini RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples: 
            #Interactive mode
            python3 main.py --interactive
            
            #Single query mode - Format: pretty
            python3 main.py --query "What movies are about AI?"
            
            #Single query mode - Format: JSON
            python3 main.py --query "What movies are about AI?" --format json
            
            #save to output file 
            python3 main.py --query "What movies are about AI?" --output result.json
            
            #run examples
            python3 main.py --examples
            
            #view configurations
            python3 main.py --show-config
        """
    )

    #mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    mode_group.add_argument(
        '--query', '-q',
        type=str,
        help='Single query to process'
    )
    mode_group.add_argument(
        '--examples', '-e',
        action='store_true',
        help='Run example queries'
    )
    mode_group.add_argument(
        '--show-config',
        action='store_true',
        help='Show current configuration and exit'
    )

    # output options
    parser.add_argument(
        '--format', '-f',
        choices=['pretty', 'json'],
        default='pretty',
        help='Output format (default: pretty)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Save output to file (JSON format)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    #handle show configs
    if args.show_config:
        print_config(config)
        sys.exit(0)

    if args.quiet:
        config.system.verbose = False


    print("\n" + "=" * 70)
    print("MOVIE PLOTS RAG SYSTEM".center(70))
    print("=" * 70)

    try:
        #initialize rag system
        rag = RAGPipeline(config)

        #show stats
        if config.system.verbose:
            stats = rag.get_stats()
            print("\n📊 System loaded:")
            print(f"  • {stats['unique_movies']} movies")
            print(f"  • {stats['total_chunks']} text chunks")
            print(f"  • {stats['embedding_dimension']}-dimensional embeddings")
            print(f"  • Model: {stats['embedding_model']}")
            print(f"  • LLM: {stats['llm_provider']}/{stats['llm_model']}")

        #select mode
        if args.interactive:
            interactive_mode(rag)
        elif args.query:
            single_query_mode(
                rag,
                args.query,
                args.format,
                args.output
            )
        elif args.examples:
            run_examples(rag, args.format)

        else:
            parser.print_help()
            print("\n" + "=" * 70)
            print("Showing examples...")
            print("=" * 70)

            response = rag.query(
                "Which movies feature artificial intelligence?",
                verbose=False
            )

            if args.format == 'json':
                print_results(response)
            else:
                print_results(response)
            print("""\n
                OPTIONS: --interactive | --query | --examples | --show-config | --help""")

    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(0)

    except Exception as e:
        print(f"\nERROR: str{e}")
        if config.system.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()