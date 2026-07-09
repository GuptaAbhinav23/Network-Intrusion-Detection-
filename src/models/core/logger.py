#hello
class Logger:

    @staticmethod
    def title(text):

        print("\n"+"="*80)
        print(text.upper())
        print("="*80)

    @staticmethod
    def info(text):

        print(text)

    @staticmethod
    def success(text):

        print(f"\n{text} Completed Successfully.")

    @staticmethod
    def error(text):

        print(text)

if __name__ == "__main__":
    Logger.title("Random Forest")

    Logger.info("Training Started")

    Logger.success("Training")