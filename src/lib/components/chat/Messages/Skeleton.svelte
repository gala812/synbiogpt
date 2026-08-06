<script lang="ts">
	export let statusText = '模型正在思考中...';
</script>

<div class="flex flex-col gap-2 mt-2 mb-2">
	<div class="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
		<div class="w-3 h-3 rounded-full border-2 border-gray-300 border-t-gray-500 animate-spin" />
		<span>{statusText}</span>
	</div>

	<div class="animate-pulse space-y-2 pt-1">
		<div class="h-2 rounded bg-gray-200 dark:bg-gray-700 w-[88%]" />
		<div class="h-2 rounded bg-gray-200 dark:bg-gray-700 w-[64%]" />
		<div class="h-2 rounded bg-gray-200 dark:bg-gray-700 w-[92%]" />
	</div>
</div>
